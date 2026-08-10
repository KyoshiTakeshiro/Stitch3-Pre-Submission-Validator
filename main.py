"""
Bitcast X pre-submission validator — FastAPI backend.

Lets a creator paste a draft tweet + pick an active campaign brief and get an
instant pass/fail verdict, replicating the same LLM evaluation logic and
optimistic multi-check strategy that real Bitcast validators use.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from prompts import generate_brief_evaluation_prompt

load_dotenv()

BITCAST_BRIEFS_ENDPOINT = "https://bitcast-api.bitcast.network/api/v2/validator/x-briefs"
BRAND_OVERVIEW_BASE_URL = "https://brand-overviews-x.s3.us-west-2.amazonaws.com"
CHUTES_ENDPOINT = "https://llm.chutes.ai/v1/chat/completions"
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
NUM_LLM_CHECKS = 3
MODEL = "Qwen/Qwen3-32B"

app = FastAPI(title="Stitch3 Validator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# /evaluate and /evaluate/stream are the only endpoints that spend real money
# (each check is a Chutes API call against our own key) -- /briefs and the
# brand-overview lookups are cheap/cached and don't need this. In-memory,
# per-process fixed-window limiter: fine for this single-instance deployment,
# same tradeoff as the existing _briefs_cache/_overview_cache dicts below.
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 5  # per IP, per window
_rate_limit_hits: dict[str, list[float]] = {}
_rate_limit_lock = Lock()


def _client_ip(request: Request) -> str:
    # Behind nginx, request.client.host is the proxy's own address unless
    # X-Forwarded-For is set -- prefer that when present.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    with _rate_limit_lock:
        hits = [t for t in _rate_limit_hits.get(ip, []) if t > cutoff]
        if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = int(hits[0] - cutoff) + 1
            raise HTTPException(
                status_code=429,
                detail="Too many checks in a short time — please wait a bit before trying again.",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)
        _rate_limit_hits[ip] = hits


class EvaluateRequest(BaseModel):
    brief_id: str
    tweet: str


class CheckResult(BaseModel):
    verdict: str
    summary: str
    raw_response: str


class EvaluateResponse(BaseModel):
    meets_brief: bool
    checks: list[CheckResult]
    prompt_version: int
    brief_display: str


def parse_verdict(text: str) -> tuple[str, str]:
    """Extract the YES/NO verdict and one-sentence summary from a model response."""
    verdict = "NO"
    summary = ""

    if "## Verdict" in text:
        after_verdict = text.split("## Verdict", 1)[1]
        verdict_line = after_verdict.strip().splitlines()[0].strip()
        if "YES" in verdict_line.upper():
            verdict = "YES"

    if "## Summary" in text:
        summary = text.split("## Summary", 1)[1].strip()
        if summary.endswith("```"):
            summary = summary[:-3].strip()

    return verdict, summary


def call_chutes(prompt: str) -> str:
    """Call Chutes' chat completions endpoint, matching the real validator's ChuteClient."""
    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 4096,
    }

    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(CHUTES_ENDPOINT, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise last_error


def run_single_check(brief: dict, tweet: str, prompt_version: int, check_num: int) -> CheckResult:
    # Real validator appends " {check_num}" to bust its LLM cache so each of the
    # NUM_LLM_CHECKS runs is an independent judgment rather than a repeat of the
    # same deterministic (temperature=0) response. Replicated here for the same
    # reason: without it, our "best of 3" carries far less real variance.
    variant_tweet = f"{tweet} {check_num}"
    prompt = generate_brief_evaluation_prompt(brief, variant_tweet, version=prompt_version)
    text = call_chutes(prompt)
    verdict, summary = parse_verdict(text)
    return CheckResult(verdict=verdict, summary=summary, raw_response=text)


def run_checks_pessimistic(brief: dict, tweet: str, prompt_version: int):
    """Yield each CheckResult as it completes, stopping at the first NO.

    Deliberately stricter than the real validator's own optimistic
    (any-YES-wins) best-of-3: a single NO here already means meets_brief
    can never be True, so there's no reason to reduce this to a numeric
    score -- the caller just needs every yielded check to be YES, and
    needs to see all NUM_LLM_CHECKS of them, for the tweet to pass. This
    makes our checker harder to satisfy than the real validator, on
    purpose -- see project notes for why (false positives here are far
    more costly to a miner than false negatives).
    """
    executor = ThreadPoolExecutor(max_workers=NUM_LLM_CHECKS)
    futures = [
        executor.submit(run_single_check, brief, tweet, prompt_version, check_num)
        for check_num in range(1, NUM_LLM_CHECKS + 1)
    ]
    try:
        for future in as_completed(futures):
            result = future.result()
            yield result
            if result.verdict != "YES":
                break
    finally:
        executor.shutdown(wait=False)


def _lookup_brief(brief_id: str) -> dict:
    resp = requests.get(BITCAST_BRIEFS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    briefs = resp.json().get("items", [])
    brief = next((b for b in briefs if b["id"] == brief_id), None)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief '{brief_id}' not found")
    return brief


def has_brand_overview(brief_id: str) -> bool:
    url = f"{BRAND_OVERVIEW_BASE_URL}/{brief_id}.pdf"
    resp = requests.head(url, timeout=10)
    return resp.status_code == 200


BRIEFS_CACHE_TTL = 300  # seconds
_briefs_cache = {"data": None, "expires_at": 0.0}
_overview_cache = {"data": None, "expires_at": 0.0}


def _get_cached_briefs() -> list[dict]:
    """Fast path: brief list only, no S3 checks. Shared by /briefs and the
    server-rendered index page, both backed by the same 5-minute cache."""
    now = time.time()
    if _briefs_cache["data"] is not None and now < _briefs_cache["expires_at"]:
        return _briefs_cache["data"]

    resp = requests.get(BITCAST_BRIEFS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])

    _briefs_cache["data"] = items
    _briefs_cache["expires_at"] = now + BRIEFS_CACHE_TTL
    return items


@app.get("/briefs")
def get_briefs():
    """has_brand_overview is filled in client-side after
    /briefs/brand-overviews resolves, so this never blocks initial page
    render on the ~1.5-3s S3 HEAD-check fan-out."""
    return _get_cached_briefs()


@app.get("/briefs/brand-overviews")
def get_brand_overviews():
    now = time.time()
    if _overview_cache["data"] is not None and now < _overview_cache["expires_at"]:
        return _overview_cache["data"]

    resp = requests.get(BITCAST_BRIEFS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    ids = [b["id"] for b in items]

    with ThreadPoolExecutor(max_workers=len(ids) or 1) as executor:
        has_overview = executor.map(has_brand_overview, ids)
    result = dict(zip(ids, has_overview))

    _overview_cache["data"] = result
    _overview_cache["expires_at"] = now + BRIEFS_CACHE_TTL
    return result


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest, request: Request):
    _enforce_rate_limit(request)
    brief = _lookup_brief(req.brief_id)
    prompt_version = brief.get("prompt_version", 1)

    checks = list(run_checks_pessimistic(brief, req.tweet, prompt_version))
    meets_brief = len(checks) == NUM_LLM_CHECKS and all(c.verdict == "YES" for c in checks)

    return EvaluateResponse(
        meets_brief=meets_brief,
        checks=checks,
        prompt_version=prompt_version,
        brief_display=brief.get("display", brief.get("brief", "")),
    )


@app.post("/evaluate/stream")
def evaluate_stream(req: EvaluateRequest, request: Request):
    """SSE variant of /evaluate: emits a "check" event as each of the (up to
    NUM_LLM_CHECKS) checks completes, then a final "done" event with the
    same shape as EvaluateResponse. Lets the frontend show real per-check
    progress during the pessimistic (must-pass-all-3) wait instead of a
    static "please wait" message."""
    _enforce_rate_limit(request)
    brief = _lookup_brief(req.brief_id)
    prompt_version = brief.get("prompt_version", 1)

    def event_stream():
        checks: list[CheckResult] = []
        for result in run_checks_pessimistic(brief, req.tweet, prompt_version):
            checks.append(result)
            payload = {"type": "check", "index": len(checks), "verdict": result.verdict}
            yield f"data: {json.dumps(payload)}\n\n"

        meets_brief = len(checks) == NUM_LLM_CHECKS and all(c.verdict == "YES" for c in checks)
        final = {
            "type": "done",
            "meets_brief": meets_brief,
            "checks": [c.model_dump() for c in checks],
            "prompt_version": prompt_version,
            "brief_display": brief.get("display", brief.get("brief", "")),
        }
        yield f"data: {json.dumps(final)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve index.html with the current briefs embedded directly in the
    response, so a first-time (uncached-browser) visitor's very first paint
    already has the data -- no separate client-side /briefs round-trip
    needed before the campaign selector/brief text can render. Falls back
    to an empty list (client JS then does its own fetch as before) if the
    Bitcast API is briefly unavailable, so a hiccup here never blocks the
    page from loading at all."""
    html = (FRONTEND_DIR / "index.html").read_text()
    try:
        briefs = _get_cached_briefs()
    except Exception:
        briefs = []

    injected = f"<script>window.__PRELOADED_BRIEFS__ = {json.dumps(briefs)};</script>"
    html = html.replace("<head>", "<head>\n" + injected, 1)
    return HTMLResponse(content=html)


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
