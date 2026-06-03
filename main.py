import asyncio
import difflib
import os
import re
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

PREDICT_MIN_LENGTH = 10
STABLE_THRESHOLD = 2
PREDICT_DEBOUNCE = 0.4

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.100.106:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


async def call_main_llm(query: str, history: list[dict] | None = None) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] API key not set in .env"
    messages = list(history or []) + [{"role": "user", "content": query}]
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.getenv("LLM_MODEL", "anthropic/claude-3.5-haiku"),
                "messages": messages,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


PREDICT_SYSTEM = """あなたはユーザーの入力途中の文章を、最も自然な完成形に予測補完するアシスタントです。
ユーザーがチャットAIに送ろうとしている質問文を、完成した1文として返してください。
余計な説明や前置きは一切不要。完成形の文章のみを出力してください。

例:
入力: 機械学習と深層学
出力: 機械学習と深層学習の違いを教えてください。

入力: Pythonで配列を昇順にソ
出力: Pythonで配列を昇順にソートする方法を教えてください。

入力: 富士山の標高は
出力: 富士山の標高は何メートルですか？"""


def is_prediction_hit(spec_query: str, actual_query: str, threshold: float = 0.75) -> bool:
    ratio = difflib.SequenceMatcher(None, spec_query, actual_query).ratio()
    print(f"[EVAL] similarity={ratio:.2f} spec={spec_query[:30]!r} actual={actual_query[:30]!r}")
    return ratio >= threshold


async def predict_completion(partial: str) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": PREDICT_SYSTEM},
                    {"role": "user", "content": f"入力: {partial}\n出力:"},
                ],
                "think": False,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 80},
            },
        )
        r.raise_for_status()
        content = r.json()["message"]["content"].strip()
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        for prefix in ("出力:", "出力：", "完成形:", "完成形："):
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
        return content


class SpeculativeSession:
    def __init__(self):
        self.current_input = ""
        self.history: list[dict] = []
        self.predict_task: asyncio.Task | None = None
        self.last_prediction: str = ""
        self.stable_count: int = 0
        self.speculative_task: asyncio.Task | None = None
        self.speculative_query: str = ""
        self.speculative_result: str | None = None
        self.real_task: asyncio.Task | None = None

    def add_to_history(self, user: str, assistant: str):
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})

    def swap_last_assistant(self, text: str):
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "assistant":
                self.history[i]["content"] = text
                break

    def cancel_predict(self):
        if self.predict_task and not self.predict_task.done():
            self.predict_task.cancel()

    def cancel_speculation(self):
        if self.speculative_task and not self.speculative_task.done():
            self.speculative_task.cancel()
        self.speculative_result = None
        self.speculative_query = ""

    def cancel_real(self):
        if self.real_task and not self.real_task.done():
            self.real_task.cancel()

    def reset_prediction(self):
        self.last_prediction = ""
        self.stable_count = 0


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = SpeculativeSession()

    async def run_speculation(query: str, history: list[dict]):
        session.speculative_query = query
        session.speculative_result = None
        await websocket.send_json({"type": "speculating", "query": query})
        print(f"[SPEC] start: {query[:40]!r}")
        try:
            result = await call_main_llm(query, history)
            session.speculative_result = result
            print(f"[SPEC] done: {len(result)} chars")
            await websocket.send_json({"type": "speculative_done", "query": query})
        except asyncio.CancelledError:
            print("[SPEC] cancelled")
        except Exception as e:
            print(f"[SPEC] error: {e}")
            session.speculative_result = None
            await websocket.send_json({"type": "speculative_error", "error": str(e)})

    async def run_prediction(text: str):
        await asyncio.sleep(PREDICT_DEBOUNCE)
        try:
            prediction = await predict_completion(text)
            print(f"[PRED] {text[:20]!r} → {prediction[:40]!r}")
            if not prediction or len(prediction) < 5:
                return
            if prediction == session.last_prediction:
                session.stable_count += 1
                print(f"[PRED] stable x{session.stable_count}")
                if session.stable_count >= STABLE_THRESHOLD:
                    spec_running = (
                        session.speculative_task is not None
                        and not session.speculative_task.done()
                    )
                    already_have = (
                        session.speculative_result is not None
                        and session.speculative_query == prediction
                    )
                    if not spec_running and not already_have:
                        session.speculative_task = asyncio.create_task(
                            run_speculation(prediction, list(session.history))
                        )
            else:
                session.last_prediction = prediction
                session.stable_count = 1
                # 予測が変わっても既存の投機結果は捨てない
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[PRED] error: {e}")

    async def run_real(actual_query: str, spec_query: str, history: list[dict]):
        print(f"[REAL] start: {actual_query[:40]!r}")
        try:
            result = await call_main_llm(actual_query, history)
            print(f"[REAL] done: {len(result)} chars")
            hit = is_prediction_hit(spec_query, actual_query)
            await websocket.send_json({
                "type": "real_response",
                "hit": hit,
                "text": result,
            })
        except asyncio.CancelledError:
            print("[REAL] cancelled")
        except Exception as e:
            print(f"[REAL] error: {e}")
            await websocket.send_json({"type": "real_error", "error": str(e)})

    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("type")

            if event == "input":
                session.current_input = data["text"]
                session.cancel_predict()

                if len(session.current_input) >= PREDICT_MIN_LENGTH:
                    session.predict_task = asyncio.create_task(run_prediction(session.current_input))
                else:
                    session.reset_prediction()
                    session.cancel_speculation()

            elif event == "submit":
                query = data["text"].strip()
                if not query:
                    continue

                session.cancel_predict()
                session.cancel_real()

                spec_result = session.speculative_result
                spec_query = session.speculative_query
                history_snapshot = list(session.history)

                if spec_result is not None:
                    await websocket.send_json({
                        "type": "response",
                        "text": spec_result,
                        "speculative_query": spec_query,
                        "cache_hit": True,
                    })
                    # 投機結果を履歴に追加（ユーザーが最初に見るのがこちら）
                    session.add_to_history(query, spec_result)
                    # 本物のプロンプトでも並行実行（履歴には入れない・比較用）
                    session.real_task = asyncio.create_task(run_real(query, spec_query, history_snapshot))
                else:
                    session.cancel_speculation()
                    await websocket.send_json({"type": "thinking"})
                    try:
                        result = await call_main_llm(query, session.history)
                        session.add_to_history(query, result)
                        await websocket.send_json({
                            "type": "response",
                            "text": result,
                            "cache_hit": False,
                        })
                    except Exception as e:
                        await websocket.send_json({"type": "error", "error": str(e)})

                session.reset_prediction()
                session.speculative_result = None
                session.speculative_query = ""

            elif event == "canon":
                session.swap_last_assistant(data["text"])
                print(f"[CANON] history updated: {data['text'][:40]!r}")

            elif event == "clear":
                session.cancel_predict()
                session.cancel_speculation()
                session.reset_prediction()
                session.current_input = ""

    except WebSocketDisconnect:
        session.cancel_predict()
        session.cancel_speculation()
        session.cancel_real()
