import asyncio
import difflib
import math
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
STABLE_THRESHOLD = 1
PREDICT_DEBOUNCE = 0.4
MAX_SPEC_CANDIDATES = 4
HIT_SIMILARITY_THRESHOLD = 0.75   # difflibフォールバック用
HIT_VECTOR_THRESHOLD = 0.75       # bge-m3 cosine類似度しきい値
STABLE_SIMILARITY = 0.50          # 前回予測との類似度がこれ以上ならstable累積（difflib・頻繁）
CANDIDATE_DEDUP_SIMILARITY = 0.92 # 既存候補とこれ以上似てたら新候補化スキップ（difflib・頻繁）

MAIN_BACKEND = os.getenv("MAIN_BACKEND", "ollama").lower()  # "ollama" | "openrouter"

MAIN_OLLAMA_BASE_URL = os.getenv("MAIN_OLLAMA_BASE_URL", "http://192.168.100.120:11434")
MAIN_OLLAMA_MODEL = os.getenv("MAIN_OLLAMA_MODEL", "qwen3.5:27b")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL", "anthropic/claude-3.5-haiku")

PREDICT_OLLAMA_BASE_URL = os.getenv("PREDICT_OLLAMA_BASE_URL", "http://192.168.100.113:11434")
PREDICT_OLLAMA_MODEL = os.getenv("PREDICT_OLLAMA_MODEL", "Qwen3:1.7b")

EMBED_OLLAMA_BASE_URL = os.getenv("EMBED_OLLAMA_BASE_URL", MAIN_OLLAMA_BASE_URL)
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


def strip_think(text: str) -> str:
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


async def call_main_llm_ollama(messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            f"{MAIN_OLLAMA_BASE_URL}/api/chat",
            json={
                "model": MAIN_OLLAMA_MODEL,
                "messages": messages,
                "think": False,
                "stream": False,
            },
        )
        r.raise_for_status()
        return strip_think(r.json()["message"]["content"])


async def call_main_llm_openrouter(messages: list[dict]) -> str:
    if not OPENROUTER_API_KEY:
        return "[ERROR] OPENROUTER_API_KEY not set"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": OPENROUTER_MODEL, "messages": messages},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def call_main_llm(query: str, history: list[dict] | None = None) -> str:
    messages = list(history or []) + [{"role": "user", "content": query}]
    if MAIN_BACKEND == "openrouter":
        return await call_main_llm_openrouter(messages)
    return await call_main_llm_ollama(messages)


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


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


async def embed(text: str) -> list[float] | None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{EMBED_OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": text},
            )
            r.raise_for_status()
            data = r.json()
            embs = data.get("embeddings") or []
            return embs[0] if embs else None
    except Exception as e:
        print(f"[EMBED] error: {e}")
        return None


def cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


async def is_prediction_hit(
    spec_query: str,
    spec_emb: list[float] | None,
    actual_query: str,
    actual_emb: list[float] | None = None,
) -> bool:
    if actual_emb is None:
        actual_emb = await embed(actual_query)
    sim_vec = cosine(spec_emb, actual_emb)
    if sim_vec is not None:
        print(f"[EVAL-vec] cos={sim_vec:.3f} spec={spec_query[:30]!r} actual={actual_query[:30]!r}")
        return sim_vec >= HIT_VECTOR_THRESHOLD
    # fallback to difflib
    ratio = similarity(spec_query, actual_query)
    print(f"[EVAL-difflib] sim={ratio:.2f} spec={spec_query[:30]!r} actual={actual_query[:30]!r}")
    return ratio >= HIT_SIMILARITY_THRESHOLD


async def predict_completion(partial: str) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{PREDICT_OLLAMA_BASE_URL}/api/chat",
            json={
                "model": PREDICT_OLLAMA_MODEL,
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
        content = strip_think(r.json()["message"]["content"].strip())
        for prefix in ("出力:", "出力:", "完成形:", "完成形:"):
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
        return content


class SpecCandidate:
    def __init__(self, query: str, history: list[dict]):
        self.query = query
        self.history = history
        self.task: asyncio.Task | None = None
        self.result: str | None = None
        self.error: bool = False
        self.embedding: list[float] | None = None


class SpeculativeSession:
    def __init__(self):
        self.current_input = ""
        self.history: list[dict] = []
        self.predict_task: asyncio.Task | None = None
        self.last_prediction: str = ""
        self.stable_count: int = 0
        self.spec_candidates: list[SpecCandidate] = []
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

    def cancel_all_speculations(self):
        for c in self.spec_candidates:
            if c.task and not c.task.done():
                c.task.cancel()
        self.spec_candidates = []

    def cancel_real(self):
        if self.real_task and not self.real_task.done():
            self.real_task.cancel()

    def reset_prediction(self):
        self.last_prediction = ""
        self.stable_count = 0

    def has_candidate_for(self, query: str) -> bool:
        return any(c.query == query for c in self.spec_candidates)

    async def best_candidate(self, actual: str):
        completed = [c for c in self.spec_candidates if c.result is not None]
        if not completed:
            return None
        actual_emb = await embed(actual)
        if actual_emb is not None and any(c.embedding for c in completed):
            scored = []
            for c in completed:
                sc = cosine(c.embedding, actual_emb)
                if sc is None:
                    sc = similarity(c.query, actual)  # difflib fallback per-candidate
                scored.append((sc, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            print(f"[BEST-vec] picks={[(round(s,3), c.query[:25]) for s,c in scored[:3]]}")
            return scored[0][1]
        # full fallback
        return max(completed, key=lambda c: similarity(c.query, actual))

    def stats(self) -> dict:
        running = sum(1 for c in self.spec_candidates if c.result is None and not c.error)
        done = sum(1 for c in self.spec_candidates if c.result is not None)
        return {"running": running, "done": done, "total": len(self.spec_candidates)}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = SpeculativeSession()

    async def _attach_embedding(candidate: SpecCandidate):
        emb = await embed(candidate.query)
        if emb is not None:
            candidate.embedding = emb

    async def run_speculation(candidate: SpecCandidate):
        print(f"[SPEC] start: {candidate.query[:40]!r}")
        await websocket.send_json({
            "type": "speculating",
            "query": candidate.query,
            **session.stats(),
        })
        try:
            result = await call_main_llm(candidate.query, candidate.history)
            candidate.result = result
            print(f"[SPEC] done ({len(result)} chars): {candidate.query[:30]!r}")
            await websocket.send_json({
                "type": "speculative_done",
                "query": candidate.query,
                **session.stats(),
            })
        except asyncio.CancelledError:
            print(f"[SPEC] cancelled: {candidate.query[:30]!r}")
        except Exception as e:
            print(f"[SPEC] error: {e}")
            candidate.error = True
            await websocket.send_json({"type": "speculative_error", "error": str(e)})

    async def run_prediction(text: str):
        await asyncio.sleep(PREDICT_DEBOUNCE)
        try:
            prediction = await predict_completion(text)
            print(f"[PRED] {text[:20]!r} -> {prediction[:40]!r}")
            if not prediction or len(prediction) < 5:
                return

            prev = session.last_prediction
            session.last_prediction = prediction
            if prev:
                sim = similarity(prev, prediction)
                if sim >= STABLE_SIMILARITY:
                    session.stable_count += 1
                    print(f"[PRED] stable x{session.stable_count} (sim={sim:.2f})")
                else:
                    session.stable_count = 1
                    print(f"[PRED] reset (sim={sim:.2f} < {STABLE_SIMILARITY})")
            else:
                session.stable_count = 1

            if session.stable_count >= STABLE_THRESHOLD:
                dup = next(
                    (c for c in session.spec_candidates
                     if similarity(prediction, c.query) >= CANDIDATE_DEDUP_SIMILARITY),
                    None,
                )
                if dup is not None:
                    print(f"[SPEC] skip dedup: similar to {dup.query[:30]!r}")
                    return
                if len(session.spec_candidates) >= MAX_SPEC_CANDIDATES:
                    print(f"[SPEC] skip: max {MAX_SPEC_CANDIDATES} candidates reached")
                    return
                candidate = SpecCandidate(prediction, list(session.history))
                session.spec_candidates.append(candidate)
                candidate.task = asyncio.create_task(run_speculation(candidate))
                asyncio.create_task(_attach_embedding(candidate))
                session.stable_count = 0  # next stable round needed for next candidate
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[PRED] error: {e}")

    async def run_real(
        actual_query: str,
        spec_query: str,
        spec_emb: list[float] | None,
        history: list[dict],
    ):
        print(f"[REAL] start: {actual_query[:40]!r}")
        try:
            result = await call_main_llm(actual_query, history)
            print(f"[REAL] done: {len(result)} chars")
            hit = await is_prediction_hit(spec_query, spec_emb, actual_query)
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
                    session.cancel_all_speculations()

            elif event == "submit":
                query = data["text"].strip()
                if not query:
                    continue

                session.cancel_predict()
                session.cancel_real()

                best = await session.best_candidate(query)
                history_snapshot = list(session.history)
                all_queries = [c.query for c in session.spec_candidates]
                picked = best.query[:30] if best else None
                print(f"[SUBMIT] {len(session.spec_candidates)} candidates -> picked: {picked!r}")

                # cancel still-running speculations
                for c in session.spec_candidates:
                    if c.task and not c.task.done():
                        c.task.cancel()

                if best is not None:
                    await websocket.send_json({
                        "type": "response",
                        "text": best.result,
                        "speculative_query": best.query,
                        "candidates": all_queries,
                        "cache_hit": True,
                    })
                    session.add_to_history(query, best.result)
                    session.real_task = asyncio.create_task(run_real(query, best.query, best.embedding, history_snapshot))
                else:
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
                session.spec_candidates = []

            elif event == "canon":
                session.swap_last_assistant(data["text"])
                print(f"[CANON] history updated: {data['text'][:40]!r}")

            elif event == "clear":
                session.cancel_predict()
                session.cancel_all_speculations()
                session.reset_prediction()
                session.current_input = ""

    except WebSocketDisconnect:
        session.cancel_predict()
        session.cancel_all_speculations()
        session.cancel_real()
