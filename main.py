from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import os
import asyncio
from datetime import datetime, timedelta
import random
import string
from openai import OpenAI
from langdetect import detect
import aiofiles
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# .envファイルから環境変数を読み込み
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # アプリケーション開始時
    cleanup_task = asyncio.create_task(cleanup_rooms())
    yield
    # アプリケーション終了時
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Translation Chat App", version="1.0.0", lifespan=lifespan)

# OpenAI クライアント（環境変数から取得）
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

openai_client = OpenAI(api_key=api_key)

# JSON データファイルのパス
ROOMS_FILE = "rooms.json"

# メモリ上でのWebSocket接続管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[Dict]] = {}  # {room_id: [{"websocket": ws, "username": user}]}

    async def connect(self, websocket: WebSocket, room_id: str, username: str = None):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        
        connection_info = {"websocket": websocket, "username": username}
        self.active_connections[room_id].append(connection_info)

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            self.active_connections[room_id] = [
                conn for conn in self.active_connections[room_id] 
                if conn["websocket"] != websocket
            ]
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

    async def broadcast_to_room(self, message: dict, room_id: str):
        """ルーム内のすべてのユーザーにメッセージを配信"""
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id].copy():
                try:
                    await connection["websocket"].send_json(message)
                except WebSocketDisconnect:
                    self.active_connections[room_id].remove(connection)

manager = ConnectionManager()

# データモデル
class MessageRequest(BaseModel):
    room_id: str
    sender: str
    message: str
    user_language: Optional[str] = 'en'

class RoomCreateRequest(BaseModel):
    room_id: Optional[str] = None

class JoinRoomRequest(BaseModel):
    username: str
    user_language: Optional[str] = 'en'

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class Message(BaseModel):
    sender: str
    original: str
    original_language: str
    translations: Dict[str, str] = {}  # 各言語への翻訳を保存
    timestamp: str

class RoomUser(BaseModel):
    username: str
    language: str
    joined_at: str

class Room(BaseModel):
    created_at: str
    users: List[RoomUser] = []
    messages: List[Message] = []

# JSON ファイル操作
async def load_rooms() -> Dict[str, Room]:
    """rooms.jsonファイルからデータを読み込む"""
    if not os.path.exists(ROOMS_FILE):
        return {}
    
    try:
        async with aiofiles.open(ROOMS_FILE, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)
            rooms = {}
            for k, v in data.get('rooms', {}).items():
                room_data = {
                    "created_at": v["created_at"],
                    "users": [RoomUser(**user) for user in v.get("users", [])],
                    "messages": [Message(**msg) for msg in v.get("messages", [])]
                }
                rooms[k] = Room(**room_data)
            return rooms
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

async def save_rooms(rooms: Dict[str, Room]):
    """rooms.jsonファイルにデータを保存"""
    data = {
        "rooms": {
            k: {
                "created_at": v.created_at,
                "users": [user.model_dump() for user in v.users],
                "messages": [msg.model_dump() for msg in v.messages]
            } for k, v in rooms.items()
        }
    }
    
    async with aiofiles.open(ROOMS_FILE, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))

def generate_room_id() -> str:
    """6桁の英数字ルームIDを生成"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

async def translate_text(text: str, user_language: str = 'en') -> tuple[str, str]:
    """テキストの言語を判定して翻訳"""
    try:
        # 言語判定
        detected_lang = detect(text)
        
        # 言語コードのマッピング
        language_names = {
            'ja': 'Japanese',
            'en': 'English', 
            'ko': 'Korean',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German'
        }
        
        # 翻訳先言語を決定（検出された言語がユーザーの主言語と異なる場合）
        if detected_lang == user_language:
            # 同じ言語の場合は英語に翻訳（または他の共通言語）
            target_lang = 'English' if user_language != 'en' else 'Japanese'
            target_code = 'en' if user_language != 'en' else 'ja'
        else:
            # 異なる言語の場合はユーザーの主言語に翻訳
            target_lang = language_names.get(user_language, 'English')
            target_code = user_language
        
        system_prompt = f"You are a translator. Translate the text to natural {target_lang}. Only return the translated text without any additional explanation."
        
        # OpenAI APIで翻訳
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        translated = response.choices[0].message.content.strip()
        return text, translated
        
    except Exception as e:
        print(f"Translation error: {e}")
        return text, f"[Translation Error] {text}"

# API エンドポイント
@app.post("/create_room")
async def create_room(request: Optional[RoomCreateRequest] = None):
    """新しいルームを作成"""
    rooms = await load_rooms()
    
    # ルームIDの生成または指定
    room_id = None
    if request and request.room_id:
        room_id = request.room_id
    else:
        room_id = generate_room_id()
    
    # 既存チェック
    if room_id in rooms:
        raise HTTPException(status_code=400, detail="Room already exists")
    
    # 新規ルーム作成
    rooms[room_id] = Room(created_at=datetime.now().isoformat())
    await save_rooms(rooms)
    
    return {"room_id": room_id, "status": "created"}

@app.post("/join_room/{room_id}")
async def join_room(room_id: str, request: JoinRoomRequest):
    """既存ルームに参加（ユーザー情報と言語設定を保存）"""
    rooms = await load_rooms()
    
    if room_id not in rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    room = rooms[room_id]
    
    # 既存ユーザーかチェック
    existing_user = None
    for user in room.users:
        if user.username == request.username:
            existing_user = user
            break
    
    if existing_user:
        # 既存ユーザーの言語設定を更新
        existing_user.language = request.user_language
    else:
        # 新規ユーザーを追加
        new_user = RoomUser(
            username=request.username,
            language=request.user_language,
            joined_at=datetime.now().isoformat()
        )
        room.users.append(new_user)
    
    await save_rooms(rooms)
    
    return {
        "room_id": room_id,
        "status": "joined",
        "message_count": len(room.messages),
        "users": [{"username": u.username, "language": u.language} for u in room.users]
    }

@app.post("/send_message")
async def send_message(request: MessageRequest):
    """メッセージを送信（全言語に翻訳して全ユーザーに配信）"""
    rooms = await load_rooms()
    
    if request.room_id not in rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    room = rooms[request.room_id]
    
    # 送信者の言語を検出
    detected_lang = detect(request.message)
    
    # ルーム内の全ユーザーの言語を取得
    user_languages = set(user.language for user in room.users)
    
    # 各言語に翻訳
    translations = {}
    for lang in user_languages:
        if lang != detected_lang:
            try:
                original, translated = await translate_text(request.message, lang)
                translations[lang] = translated
            except Exception as e:
                print(f"Translation error for {lang}: {e}")
                translations[lang] = request.message
        else:
            # 原文と同じ言語の場合はそのまま
            translations[lang] = request.message
    
    # メッセージオブジェクトを作成
    message = Message(
        sender=request.sender,
        original=request.message,
        original_language=detected_lang,
        translations=translations,
        timestamp=datetime.now().isoformat()
    )
    
    # ルームにメッセージを保存
    room.messages.append(message)
    await save_rooms(rooms)
    
    # WebSocket経由で全ユーザーに同じメッセージを配信
    await manager.broadcast_to_room(message.model_dump(), request.room_id)
    
    return {"status": "sent", "message": message.model_dump()}

@app.get("/get_messages/{room_id}")
async def get_messages(room_id: str):
    """ルームの全チャット履歴を取得"""
    rooms = await load_rooms()
    
    if room_id not in rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    return {"room_id": room_id, "messages": [msg.model_dump() for msg in rooms[room_id].messages]}

@app.websocket("/ws/{room_id}/{username}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, username: str):
    """WebSocketでリアルタイム通信（ユーザー名付き）"""
    await manager.connect(websocket, room_id, username)
    try:
        while True:
            # クライアントからのメッセージを待機（Keep-Alive用）
            data = await websocket.receive_text()
            # 必要に応じてハートビート処理など
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)

# 定期的なルーム掃除（30分無操作で削除）
async def cleanup_rooms():
    """30分以上古いルームを削除"""
    while True:
        try:
            rooms = await load_rooms()
            now = datetime.now()
            rooms_to_delete = []
            
            for room_id, room in rooms.items():
                created_time = datetime.fromisoformat(room.created_at)
                if now - created_time > timedelta(minutes=30):
                    rooms_to_delete.append(room_id)
            
            for room_id in rooms_to_delete:
                del rooms[room_id]
            
            if rooms_to_delete:
                await save_rooms(rooms)
                print(f"Cleaned up {len(rooms_to_delete)} rooms")
                
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        # 10分おきにチェック
        await asyncio.sleep(600)

# 静的ファイル配信（フロントエンド用）
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/translator")
async def read_translator():
    return FileResponse('static/translator.html')

@app.post("/api/translate")
async def translate_text_api(request: TranslateRequest):
    """翻訳ツール用のAPIエンドポイント"""
    try:
        # 言語コードのマッピング
        language_names = {
            'ja': 'Japanese',
            'en': 'English', 
            'ko': 'Korean',
            'zh': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German'
        }
        
        source_lang_name = language_names.get(request.source_lang, 'English')
        target_lang_name = language_names.get(request.target_lang, 'English')
        
        # 同じ言語の場合は翻訳不要
        if request.source_lang == request.target_lang:
            return {"translated_text": request.text, "source_lang": request.source_lang, "target_lang": request.target_lang}
        
        system_prompt = f"You are a translator. Translate the text from {source_lang_name} to natural {target_lang_name}. Only return the translated text without any additional explanation."
        
        # OpenAI APIで翻訳
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        translated = response.choices[0].message.content.strip()
        
        return {
            "translated_text": translated,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang
        }
        
    except Exception as e:
        print(f"Translation API error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 