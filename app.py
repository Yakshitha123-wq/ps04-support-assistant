import json
import os
import re

import numpy as np
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Create a .env file locally with "
        "GEMINI_API_KEY=your_key_here (never commit this file)."
    )

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = """TRACK_ID=PS04

You are a resolution assistant for a broadband and mobile provider's support desk.

You are given three things for every request:
1. The conversation so far (what the customer has said)
2. The customer's account record (plan, billing status, recent tickets)
3. A set of support articles

Decide ONE of three outcomes:

1. RESPOND - if the request is routine and a support article clearly covers it,
   AND you have everything you need to give an actual resolution (not just a
   question). If your answer would just be a question back to the customer,
   that is NOT a respond - use ASK_FOR_INFO instead. Never put a clarifying
   question inside an "answer" field.
   Draft a resolution grounded in that article. Always cite which article you used.
   If you draw on more than one article (e.g. resolving multiple issues in one
   message), list every article you actually used in the citation field, separated
   by commas - never cite only one when more than one was used.
   Never answer from anything other than the provided article text.

2. ASK_FOR_INFO - if you cannot proceed without something specific from the customer
   (e.g. an account number, a date, confirmation of an issue, or details an article
   says to ask for before resolving - e.g. "ask whether it happens on all devices").
   Ask for exactly what you need - nothing vague. This is the correct decision
   whenever your response is itself a question, even if it cites an article.

3. ESCALATE - if the case is complex, uncertain, or not covered by any article.
   Hand over a concise summary so the human agent does not make the customer repeat
   themselves: what the issue is, what has been established so far, and what (if
   anything) has already been tried.

Never guess or invent policy, article content, or account details that are not
explicitly given to you.

EDGE CASES you must handle:
- If the customer's message is not actually a support issue (e.g. a thank-you,
  small talk, confirmation that something is already resolved), use RESPOND with
  a brief acknowledgment. Do not force a citation, and do not escalate something
  that requires no action.
- If the message raises more than one issue, evaluate each one individually. If every
  issue raised can be resolved directly from the account record and support articles,
  use RESPOND and address each one clearly in your answer - do not escalate just
  because multiple things were mentioned. Only escalate the whole message if at least
  one of the issues genuinely requires human judgment, is ambiguous, or isn't covered
  by any article.
- If what the customer describes contradicts the account record (e.g. they claim
  a plan or charge that doesn't match what's on file), do not resolve based on
  the customer's claim alone. Use ASK_FOR_INFO to clarify, or ESCALATE if the
  discrepancy itself is the issue.
- When writing the "established" field, be precise about what is actually confirmed
  by the account record versus what is only the customer's claim. If the account
  record has no data relevant to the claim (e.g. no installation ticket at all),
  say so explicitly ("customer claims X; no related record found") rather than
  restating the claim as if it were a confirmed fact. If the account record
  contradicts the claim, state the contradiction explicitly.
- Check whether the account record actually has the type of service being asked
  about (e.g. broadband vs mobile). If the customer asks about a service type
  their account record does not show (e.g. a mobile-only account asked about
  wifi speed), explicitly note this mismatch rather than proceeding as if the
  service exists.
- If no account record is found for the given customer, the "established" or
  "issue" field must explicitly state that no account record exists for that
  customer ID - never omit this or imply the account was checked normally.
- If the account record shows an existing open or recent ticket relevant to this
  request, factor that history into your decision - a second occurrence of an
  already-"resolved" issue is a signal to escalate, not repeat the same fix.
- If no account record is found for the given customer, ESCALATE - never proceed
  without a valid account record.

Always respond in exactly one of these JSON shapes and nothing else:

{"decision": "respond", "answer": "...", "citation": "article title or id"}

{"decision": "ask_for_info", "question": "..."}

{"decision": "escalate", "handover_summary": {"issue": "...", "established": "...", "already_tried": "..."}}
"""


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_articles(raw_text: str) -> list:
    articles = []
    parts = re.split(r"(?=### Article )", raw_text)
    for part in parts:
        part = part.strip()
        if not part.startswith("### Article"):
            continue
        lines = part.split("\n")
        title_line = lines[0].replace("### ", "").strip()
        body = "\n".join(lines[1:]).strip()
        articles.append({"title": title_line, "text": body})
    return articles


def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_text(text: str) -> list:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return result.embeddings[0].values


SUPPORT_ARTICLES_RAW = load_text_file(os.path.join("data", "support_articles.md"))
ARTICLES = parse_articles(SUPPORT_ARTICLES_RAW)
ALL_ACCOUNT_RECORDS = load_text_file(os.path.join("data", "account_records.md"))

EMBEDDINGS_PATH = os.path.join("data", "article_embeddings.json")


def load_or_build_article_embeddings() -> list:
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(ARTICLES):
            return [item["embedding"] for item in cached]

    embeddings = []
    for article in ARTICLES:
        embeddings.append(embed_text(article["text"]))

    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"title": a["title"], "embedding": e} for a, e in zip(ARTICLES, embeddings)],
            f,
        )
    return embeddings


ARTICLE_EMBEDDINGS = load_or_build_article_embeddings()


def split_into_segments(conversation: str) -> list:
    parts = re.split(r"(?:,?\s+also\s+|;|\n)", conversation, flags=re.IGNORECASE)
    segments = []
    for part in parts:
        subparts = re.split(r"(?<=[.?!])\s+", part.strip())
        segments.extend([s.strip() for s in subparts if len(s.strip()) > 5])
    return segments if segments else [conversation]


def retrieve_relevant_articles(conversation: str, top_k_per_segment: int = 2, max_total: int = 5) -> str:
    segments = split_into_segments(conversation)

    seen_titles = set()
    collected = []

    for segment in segments:
        segment_embedding = embed_text(segment)
        scored = []
        for article, embedding in zip(ARTICLES, ARTICLE_EMBEDDINGS):
            score = cosine_similarity(segment_embedding, embedding)
            scored.append((score, article))
        scored.sort(key=lambda x: x[0], reverse=True)

        for score, article in scored[:top_k_per_segment]:
            if article["title"] not in seen_titles:
                seen_titles.add(article["title"])
                collected.append(article)

    collected = collected[:max_total]
    return "\n\n".join(f"{a['title']}\n{a['text']}" for a in collected)


def get_account_record(customer_id: str) -> str:
    blocks = ALL_ACCOUNT_RECORDS.split("## Customer:")
    for block in blocks[1:]:
        if customer_id in block.split("\n")[0]:
            return "## Customer:" + block
    return f"No account record found for customer ID '{customer_id}'."


def call_agent(conversation: str, customer_id: str) -> str:
    account_record = get_account_record(customer_id)
    relevant_articles = retrieve_relevant_articles(conversation)
    prompt = (
        f"Account record:\n{account_record}\n\n"
        f"Most relevant support articles for this conversation:\n{relevant_articles}\n\n"
        f"Conversation so far:\n{conversation}"
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={"system_instruction": SYSTEM_PROMPT},
    )
    return response.text


def parse_agent_response(raw_text: str) -> dict:
    try:
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "decision": "escalate",
            "handover_summary": {
                "issue": "Could not parse model response - failing safe.",
                "established": "N/A",
                "already_tried": "N/A",
            },
        }

    if data.get("decision") not in ("respond", "ask_for_info", "escalate"):
        return {
            "decision": "escalate",
            "handover_summary": {
                "issue": "Model returned an unexpected decision value.",
                "established": "N/A",
                "already_tried": "N/A",
            },
        }

    if data.get("decision") == "respond":
        answer_text = data.get("answer", "").strip()
        if answer_text.endswith("?"):
            return {
                "decision": "ask_for_info",
                "question": answer_text,
            }

    return data


app = FastAPI()


class TicketRequest(BaseModel):
    conversation: str
    customer_id: str


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/handle_ticket")
def handle_ticket(req: TicketRequest):
    if not req.conversation or not req.conversation.strip():
        return {
            "decision": "escalate",
            "handover_summary": {
                "issue": "Empty or missing conversation text.",
                "established": "N/A",
                "already_tried": "N/A",
            },
        }

    try:
        raw = call_agent(req.conversation, req.customer_id)
    except Exception as e:
        return {
            "decision": "escalate",
            "handover_summary": {
                "issue": "Model call failed.",
                "established": str(e),
                "already_tried": "N/A",
            },
        }
    return parse_agent_response(raw)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)