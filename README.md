# Speculative Chat

タイピング中にローカルLLMがプロンプトを予測し、確度が高まったらメインLLMに先行投げする**投機的実行チャット**のプロトタイプ。

## 仕組み

```
ユーザー入力（途中）
    ↓
ローカルLLM（Ollama）がプロンプト補完を予測
    ↓ 予測が2回連続で安定したら
メインLLM（OpenRouter）に投機的に先行リクエスト
    ↓ ユーザーが送信
投機結果を即座に表示 ＋ 本物のプロンプトでも並行実行
    ↓
「予測プロンプト」と「実際のプロンプト」の回答を切替可能
```

## 特徴

- **即レスポンス**: 入力中に投機実行が完了していれば、送信と同時に回答が表示される
- **透明性**: 予測されたプロンプトをUIに表示し、ユーザーが判断できる
- **切替可能**: 投機回答と本物の回答をワンクリックで比較できる

## セットアップ

```bash
git clone https://github.com/kamarogit/speculative-chat
cd speculative-chat
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env を編集してAPIキーとモデルを設定
```

## 環境変数（.env）

```env
# メインLLM（OpenRouter推奨）
OPENROUTER_API_KEY=your_key_here
LLM_MODEL=anthropic/claude-3.5-haiku

# プロンプト予測用ローカルLLM（Ollama）
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
```

## 起動

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

ブラウザで `http://localhost:8000` を開く。

## 動作のコツ

- **10文字以上**入力するとOllamaによる予測が始まる
- 予測が**2回連続で一致**すると投機的実行がトリガーされる
- ステータスバーがオレンジ→緑に変わったら投機実行完了
- 送信すると投機回答（⚡）が即表示され、本物の回答が並行生成される

## 今後の方向性

- 投機クエリと実際のクエリのベクトル距離による自動マッチング
- ストリーミングレスポンス対応
- 投機ヒット率の統計表示
