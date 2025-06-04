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

# .envファイルから環境変数を読み込み
load_dotenv()

app = FastAPI(title="Translation Chat App", version="1.0.0")

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
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

    async def broadcast(self, message: dict, room_id: str):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id].copy():
                try:
                    await connection.send_json(message)
                except WebSocketDisconnect:
                    self.active_connections[room_id].remove(connection)

manager = ConnectionManager()

# データモデル
class MessageRequest(BaseModel):
    room_id: str
    sender: str
    message: str

class RoomCreateRequest(BaseModel):
    room_id: Optional[str] = None

class Message(BaseModel):
    sender: str
    original: str
    translated: str
    timestamp: str

class Room(BaseModel):
    created_at: str
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
            return {k: Room(**v) for k, v in data.get('rooms', {}).items()}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

async def save_rooms(rooms: Dict[str, Room]):
    """rooms.jsonファイルにデータを保存"""
    data = {
        "rooms": {
            k: {
                "created_at": v.created_at,
                "messages": [msg.dict() for msg in v.messages]
            } for k, v in rooms.items()
        }
    }
    
    async with aiofiles.open(ROOMS_FILE, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))

def generate_room_id() -> str:
    """6桁の英数字ルームIDを生成"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

async def translate_text(text: str) -> tuple[str, str]:
    """テキストの言語を判定して翻訳"""
    try:
        # 言語判定
        detected_lang = detect(text)
        
        # 翻訳先言語を決定
        if detected_lang == 'ja':
            target_lang = "English"
            system_prompt = "You are a translator. Translate the Japanese text to natural English."
        else:
            target_lang = "Japanese"
            system_prompt = "You are a translator. Translate the text to natural Japanese."
        
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
        return text, f"[翻訳エラー] {text}"

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

@app.get("/join_room/{room_id}")
async def join_room(room_id: str):
    """既存ルームに参加可能かチェック"""
    rooms = await load_rooms()
    
    if room_id not in rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    return {"room_id": room_id, "status": "available", "message_count": len(rooms[room_id].messages)}

@app.post("/send_message")
async def send_message(request: MessageRequest):
    """メッセージを送信（翻訳付き）"""
    rooms = await load_rooms()
    
    if request.room_id not in rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    # 翻訳実行
    original, translated = await translate_text(request.message)
    
    # メッセージ作成
    message = Message(
        sender=request.sender,
        original=original,
        translated=translated,
        timestamp=datetime.now().isoformat()
    )
    
    # ルームに追加
    rooms[request.room_id].messages.append(message)
    await save_rooms(rooms)
    
    # WebSocket経由でブロードキャスト
    await manager.broadcast(message.dict(), request.room_id)
    
    return {"status": "sent", "message": message.dict()}

@app.get("/get_messages/{room_id}")
async def get_messages(room_id: str):
    """ルームの全チャット履歴を取得"""
    rooms = await load_rooms()
    
    if room_id not in rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    return {"room_id": room_id, "messages": [msg.dict() for msg in rooms[room_id].messages]}

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    """WebSocketでリアルタイム通信"""
    await manager.connect(websocket, room_id)
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

# アプリケーション起動時にクリーンアップタスクを開始
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_rooms())

# 静的ファイル配信（フロントエンド用）
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 