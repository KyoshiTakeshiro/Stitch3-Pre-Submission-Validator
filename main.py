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

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
def evaluate(req: EvaluateRequest):
    resp = requests.get(BITCAST_BRIEFS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    briefs = resp.json().get("items", [])

    brief = next((b for b in briefs if b["id"] == req.brief_id), None)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief '{req.brief_id}' not found")

    prompt_version = brief.get("prompt_version", 1)

    executor = ThreadPoolExecutor(max_workers=NUM_LLM_CHECKS)
    futures = [
        executor.submit(run_single_check, brief, req.tweet, prompt_version, check_num)
        for check_num in range(1, NUM_LLM_CHECKS + 1)
    ]

    checks: list[CheckResult] = []
    meets_brief = False
    for future in as_completed(futures):
        checks.append(future.result())
        if checks[-1].verdict == "YES":
            meets_brief = True
            break

    # Any remaining checks (if we exited early on a YES) finish in the
    # background and are simply discarded — no need to block the response.
    executor.shutdown(wait=False)

    return EvaluateResponse(
        meets_brief=meets_brief,
        checks=checks,
        prompt_version=prompt_version,
        brief_display=brief.get("display", brief.get("brief", "")),
    )


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
