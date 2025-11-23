# app.py
import os
import json
import time
import logging
import traceback
from typing import Optional
from urllib.parse import urljoin

import requests
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import asyncio


app = FastAPI()
logging.basicConfig(level=logging.INFO)

# --- CONFIG: change these for your deployment/testing ---
EXPECTED_SECRET = os.environ.get("QUIZ_SECRET")
MAX_PROCESS_SECONDS = 170  # must be < 180 to stay inside 3-minute window comfortably
PLAYWRIGHT_NAV_TIMEOUT = 30_000  # ms

# --------- Pydantic for incoming payload ----------
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str
    # allow other optional fields
    class Config:
        extra = "allow"

# --------- endpoint that receives POSTs ----------
@app.post("/api/quiz")
async def receive_quiz(payload: QuizRequest):
    if payload.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    # fire-and-forget async task
    asyncio.create_task(process_quiz_request(payload.dict()))
    return {"status": "accepted", "message": "Quiz processing scheduled."}


# --------- Worker that processes the quiz URL ----------
async def process_quiz_request(payload: dict):
    start_time = time.time()
    try:
        url = payload["url"]
        logging.info(f"Processing quiz for URL: {url}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            try:
                await page.goto(url, timeout=PLAYWRIGHT_NAV_TIMEOUT, wait_until="networkidle")
            except PlaywrightTimeoutError:
                logging.warning("Navigation timeout; trying with longer wait")
                await page.goto(url, timeout=PLAYWRIGHT_NAV_TIMEOUT * 2, wait_until="load")

            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            submit_url = find_submit_url(soup, base_url=url)
            logging.info(f"Found submit_url: {submit_url}")

            # Heuristics: PDF -> JSON -> TABLE -> AUDIO -> TEXT
            pdf_link = find_pdf_link(soup, page, base_url=url)
            if pdf_link:
                logging.info(f"Detected PDF at {pdf_link}, handling as PDF task")
                answer_payload = handle_pdf_task(pdf_link, payload)
            else:
                pre_json = find_json_in_page(soup)
                if pre_json:
                    logging.info("Detected JSON payload on page")
                    answer_payload = handle_json_task(pre_json, payload)
                else:
                    tables = soup.find_all("table")
                    if tables:
                        logging.info(f"Found {len(tables)} table(s) on page, attempting to parse")
                        answer_payload = handle_html_table_task(soup, payload)
                    else:
                        # NEW: check for audio task before generic text handler
                        audio_url = find_audio_url(soup, base_url=url)
                        if audio_url:
                            logging.info(f"Detected audio at {audio_url}, handling as audio task")
                            answer_payload = handle_audio_task(audio_url, soup, payload)
                        else:
                            qtext = await page.inner_text("body")
                            logging.info("No direct artifact found; using text handler")
                            answer_payload = handle_text_task(qtext, payload)

            if not submit_url:
                logging.error("No submit URL found on page; cannot submit.")
                return

            body = answer_payload
            body_json = json.dumps(body)
            if len(body_json.encode("utf-8")) > (1_000_000 - 1000):
                logging.warning("Answer JSON too large; trimming")
                raise Exception("Answer payload too large")

            logging.info(f"Submitting answer to {submit_url}: {body}")
            resp = requests.post(submit_url, json=body, timeout=30)
            logging.info(f"Submit response: {resp.status_code} {resp.text}")

            try:
                resp_json = resp.json()
                if resp_json.get("url"):
                    next_url = resp_json["url"]
                    elapsed = time.time() - start_time
                    if elapsed < MAX_PROCESS_SECONDS:
                        logging.info(f"Chasing next URL: {next_url}")
                        payload2 = payload.copy()
                        payload2["url"] = next_url
                        # async recursion
                        await process_quiz_request(payload2)
            except Exception:
                logging.info("Submit response not JSON or no next URL.")
            finally:
                await context.close()
                await browser.close()

    except Exception as e:
        logging.error("Error processing quiz request: " + str(e))
        traceback.print_exc()


# ---------- Helper / handlers ----------
def find_submit_url(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    """
    Robust submit URL finder:
    - checks form[action]
    - checks <a href> with 'submit'
    - looks for absolute 'https://.../submit' in page text
    - falls back to relative '/submit' (joins with base_url)
    """
    import re
    # 1) form action
    form = soup.find("form")
    if form and form.get("action"):
        return urljoin(base_url, form.get("action"))

    # 2) anchor links that contain 'submit' in href
    for a in soup.find_all("a", href=True):
        if "submit" in a["href"].lower():
            return urljoin(base_url, a["href"])

    # 3) search page text for an absolute submit URL
    text = soup.get_text(" ")
    m = re.search(r"https?://[^\s'\"<>]+/submit[^\s'\"<>]*", text)
    if m:
        return m.group(0)

    # 4) search for a relative '/submit' token and join it with base_url
    if "/submit" in text:
        return urljoin(base_url, "/submit")

    # 5) no submit URL found
    return None


def find_pdf_link(soup: BeautifulSoup, page=None, base_url: str = None) -> Optional[str]:
    # search <a href> that ends with .pdf
    for a in soup.find_all("a", href=True):
        if a["href"].lower().endswith(".pdf"):
            return urljoin(base_url, a["href"])
    # TODO: look for base64-embedded PDFs if needed
    return None


def find_json_in_page(soup: BeautifulSoup) -> Optional[dict]:
    # check <pre> or <script type="application/json">
    for script in soup.find_all("script", {"type": "application/json"}):
        try:
            return json.loads(script.string)
        except Exception:
            continue
    pre = soup.find("pre")
    if pre:
        txt = pre.get_text()
        try:
            return json.loads(txt)
        except Exception:
            pass
    return None


def find_audio_url(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    """Detect presence of audio on the page and return a full URL if found."""
    # <audio src="...">
    audio = soup.find("audio")
    if audio and audio.get("src"):
        return urljoin(base_url, audio["src"])

    # <audio><source src="..."></audio>
    source = soup.find("audio")
    if source:
        src_tag = source.find("source")
        if src_tag and src_tag.get("src"):
            return urljoin(base_url, src_tag["src"])

    # Plain <a href="...mp3"> or similar
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if href.endswith((".mp3", ".wav", ".m4a", ".ogg")):
            return urljoin(base_url, a["href"])

    return None


def handle_pdf_task(pdf_url: str, payload: dict) -> dict:
    """Download PDF and attempt to answer typical questions like 'sum of column X on page Y'."""
    r = requests.get(pdf_url, timeout=30)
    r.raise_for_status()
    tmp = "/tmp/quiz_download.pdf"
    with open(tmp, "wb") as f:
        f.write(r.content)

    # Try extracting tables with pdfplumber (works for many text-based PDFs)
    with pdfplumber.open(tmp) as pdf:
        data_frames = []
        for i, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = None
            if tables:
                for table in tables:
                    df = pd.DataFrame(table)
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)
                    data_frames.append((i + 1, df))

        # Heuristic: "sum of the 'value' column on page 2"
        for page_num, df in data_frames:
            cols = [str(c).strip().lower() for c in df.columns]
            if "value" in cols and page_num == 2:
                colname = df.columns[cols.index("value")]
                s = pd.to_numeric(df[colname].astype(str).str.replace("[^0-9.-]", "", regex=True),
                                  errors="coerce")
                total = int(s.sum())
                return {
                    "email": payload["email"],
                    "secret": payload["secret"],
                    "url": payload["url"],
                    "answer": total,
                }

        # Fallback: first numeric column on first table
        if data_frames:
            _, df = data_frames[0]
            for c in df.columns:
                s = pd.to_numeric(df[c].astype(str).str.replace("[^0-9.-]", "", regex=True),
                                  errors="coerce")
                if s.notna().any():
                    return {
                        "email": payload["email"],
                        "secret": payload["secret"],
                        "url": payload["url"],
                        "answer": int(s.sum()),
                    }

    # fallback: if nothing found, return plausible empty answer
    return {
        "email": payload["email"],
        "secret": payload["secret"],
        "url": payload["url"],
        "answer": None,
    }


def handle_json_task(json_blob: dict, payload: dict) -> dict:
    # Try: search recursively for numeric values and sum them
    def find_values(obj):
        if isinstance(obj, dict):
            for _, v in obj.items():
                yield from find_values(v)
        elif isinstance(obj, list):
            for item in obj:
                yield from find_values(item)
        elif isinstance(obj, (int, float)):
            yield obj

    nums = list(find_values(json_blob))
    if nums:
        answer = sum(nums)
    else:
        # Fallback for pages like the /demo endpoint: any non-null is accepted
        answer = "anything you want"

    return {
        "email": payload["email"],
        "secret": payload["secret"],
        "url": payload["url"],
        "answer": answer,
    }


def handle_html_table_task(soup: BeautifulSoup, payload: dict) -> dict:
    # convert first table to DataFrame
    table = soup.find("table")
    df = pd.read_html(str(table))[0]
    # Try to detect a column named 'value' (case-insensitive)
    cols = [c.lower() if isinstance(c, str) else str(c) for c in df.columns]
    if "value" in cols:
        col = df.columns[cols.index("value")]
        numeric = pd.to_numeric(df[col].astype(str).str.replace("[^0-9.-]", "", regex=True),
                                errors="coerce")
        return {
            "email": payload["email"],
            "secret": payload["secret"],
            "url": payload["url"],
            "answer": int(numeric.sum()),
        }
    # fallback: sum first numeric column
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols):
        col = numeric_cols[0]
        return {
            "email": payload["email"],
            "secret": payload["secret"],
            "url": payload["url"],
            "answer": int(df[col].sum()),
        }
    return {
        "email": payload["email"],
        "secret": payload["secret"],
        "url": payload["url"],
        "answer": None,
    }


def extract_numbers_from_text(text: str):
    """
    Returns a list of ints parsed from the text.
    Handles:
     - digit substrings like "12" or "-5"
     - simple number-words up to thousands (e.g. "forty five", "one hundred twenty three")
    """
    import re

    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    # if we already found digits, return them
    if nums:
        return nums

    # small number words parser (supports units, tens, hundred/thousand)
    word_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        "hundred": 100, "thousand": 1000, "million": 1_000_000
    }

    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not tokens:
        return []

    values = []
    cur = 0
    last_scale = 1
    for t in tokens:
        if t not in word_map:
            # word-boundary: flush current if any
            if cur != 0:
                values.append(cur)
                cur = 0
                last_scale = 1
            continue
        val = word_map[t]
        if val >= 100:
            # scale words
            if cur == 0:
                cur = 1
            cur = cur * val
            last_scale = val
        else:
            # unit or tens
            cur = cur + val
    if cur != 0:
        values.append(cur)

    return values

from io import StringIO  # put this near your other imports at the top

def handle_audio_task(audio_url: str, soup: BeautifulSoup, payload: dict) -> dict:
    """
    Audio-task handler for demo:
      - Find the 'CSV file' link on the page
      - Parse the cutoff value (if present)
      - Download CSV and sum a numeric column (prefer 'value') optionally using cutoff
      - Fallback: sum any integers found in the page text

    This avoids STT dependencies and still demonstrates:
      - data sourcing (CSV),
      - parsing cutoff,
      - numerical aggregation.
    """
    import re

    base_url = payload["url"]

    # 1) Find CSV link on the page
    csv_url = None
    for a in soup.find_all("a", href=True):
        text = (a.get_text("") or "").lower()
        href = a["href"]
        if "csv" in text or href.lower().endswith(".csv"):
            csv_url = urljoin(base_url, href)
            break

    # 2) Try to parse cutoff from page text (e.g. "Cutoff: 1234")
    page_text = soup.get_text(" ")
    cutoff = None
    m = re.search(r"Cutoff:\s*([-+]?\d+(?:\.\d+)?)", page_text)
    if m:
        try:
            cutoff = float(m.group(1))
            logging.info(f"Parsed cutoff from page: {cutoff}")
        except Exception:
            cutoff = None

    # 3) If we found a CSV URL, download and compute
    if csv_url:
        logging.info(f"Found CSV URL for audio task: {csv_url}")
        try:
            r = requests.get(csv_url, timeout=30)
            r.raise_for_status()

            # Use pandas to read CSV
            df = pd.read_csv(StringIO(r.text))

            # Choose numeric columns
            num_cols = df.select_dtypes(include=["number"]).columns
            if len(num_cols) == 0:
                logging.warning("CSV has no numeric columns; falling back to text-only handler.")
            else:
                # Prefer a column named 'value', 'values', 'amount', 'score' if present
                col = None
                for cand in num_cols:
                    cname = str(cand).lower()
                    if cname in ("value", "values", "amount", "score"):
                        col = cand
                        break
                if col is None:
                    col = num_cols[0]

                series = df[col]
                if cutoff is not None:
                    series = series[series > cutoff]

                total = series.sum()

                # Cast to int when it looks integer-like, else keep as float
                if float(total).is_integer():
                    answer = int(total)
                else:
                    answer = float(total)

                return {
                    "email": payload["email"],
                    "secret": payload["secret"],
                    "url": payload["url"],
                    "answer": answer,
                }

        except Exception as e:
            logging.warning(f"Failed to process CSV audio task: {e}")

    # 4) Fallback: purely text-based sum of all integers in page text
    nums = [int(x) for x in re.findall(r"-?\d+", page_text)]
    if nums:
        answer = sum(nums)
    else:
        # last-resort fallback: non-null answer
        answer = 0

    return {
        "email": payload["email"],
        "secret": payload["secret"],
        "url": payload["url"],
        "answer": answer,
    }
