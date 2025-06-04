# 🗣️ 翻訳チャットアプリ

訪日外国人と日本人が翻訳しながらリアルタイムでチャットできるWebアプリケーションです。

## ✨ 特徴

- **面倒な登録不要**: ルームIDを共有するだけで即座にチャット開始
- **リアルタイム翻訳**: OpenAI GPT-4による高精度な翻訳
- **双方向対応**: 日本語 ⇄ 英語の自動判定・翻訳
- **WebSocket通信**: リアルタイムでメッセージを同期
- **シンプルなUI**: 直感的で使いやすいチャットインターフェース

## 🛠️ 技術スタック

- **バックエンド**: FastAPI + Python
- **フロントエンド**: Vanilla HTML/CSS/JavaScript
- **データ管理**: JSON ファイル (rooms.json)
- **翻訳API**: OpenAI GPT-4
- **リアルタイム通信**: WebSocket

## 📋 必要要件

- Python 3.8+
- OpenAI API キー

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
# 依存関係をインストール
pip install -r requirements.txt
```

### 2. 環境変数の設定（.envファイル）

プロジェクトルートに `.env` ファイルを作成し、以下の内容を設定してください：

```bash
# .env ファイルの内容
OPENAI_API_KEY=your-openai-api-key-here
```

**.envファイルの作成例：**
```bash
# .envファイルを作成
echo "OPENAI_API_KEY=your-actual-api-key-here" > .env
```

**注意**: `.env`ファイルには実際のAPIキーが含まれるため、Gitにコミットしないよう注意してください。

### 3. アプリケーションの起動

```bash
# 開発サーバーを起動
python main.py

# または uvicorn を直接使用
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. アプリケーションにアクセス

ブラウザで `http://localhost:8000` にアクセスしてください。

## 📚 API エンドポイント

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| `POST` | `/create_room` | 新しいチャットルームを作成 |
| `GET` | `/join_room/{room_id}` | 既存ルームの参加可能性をチェック |
| `POST` | `/send_message` | メッセージの送信と翻訳 |
| `GET` | `/get_messages/{room_id}` | ルームのチャット履歴を取得 |
| `WebSocket` | `/ws/{room_id}` | リアルタイム通信 |

## 💡 使い方

1. **ルーム作成**: 「新しいルーム作成」ボタンで6桁のルームIDを生成
2. **ルーム共有**: 生成されたルームIDを相手に共有
3. **参加**: 相手はルームIDを入力して同じルームに参加
4. **チャット開始**: メッセージを入力すると自動で翻訳されて相手に届く

## 🔧 設定

### ルームの有効期限

- デフォルト: 30分間無操作で自動削除
- `main.py` の `cleanup_rooms()` 関数で変更可能

### 翻訳設定

- 対応言語: 日本語 ⇄ 英語（自動判定）
- 拡張可能: `translate_text()` 関数で他言語にも対応可

## 📁 プロジェクト構造

```
translation_app/
├── main.py              # FastAPI メインアプリケーション
├── requirements.txt     # Python 依存関係
├── README.md           # このファイル
├── static/
│   └── index.html      # フロントエンド HTML
└── rooms.json          # チャットデータ（自動生成）
```

## 🔒 注意事項

- **OpenAI API料金**: 翻訳にOpenAI APIを使用するため、使用量に応じて料金が発生します
- **データ永続化**: 現在はJSONファイルでデータを管理しているため、サーバー再起動時にデータが保持されます
- **スケーラビリティ**: 大規模な利用には Redis や本格的なデータベースへの移行を推奨

## 🚀 本番環境での運用

### Docker での運用

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 環境変数

本番環境では以下の環境変数を設定してください：

- `OPENAI_API_KEY`: OpenAI API キー
- `HOST`: サーバーホスト（デフォルト: 0.0.0.0）
- `PORT`: ポート番号（デフォルト: 8000）

## 🤝 開発に参加

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 🙋‍♂️ サポート

質問や問題がある場合は、Issue を作成してください。 