import json
import os

from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI
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

SYSTEM_PROMPT = """You are a resolution assistant for a broadband and mobile provider's support desk.

You are given three things for every request:
1. The conversation so far (what the customer has said)
2. The customer's account record (plan, billing status, recent tickets)
3. A set of support articles

Decide ONE of three outcomes:

1. RESPOND - if the request is routine and a support article clearly covers it.
   Draft a resolution grounded in that article. Always cite which article you used.
   Never answer from anything other than the provided article text.

2. ASK_FOR_INFO - if you cannot proceed without something specific from the customer
   (e.g. an account number, a date, confirmation of an issue). Ask for exactly what
   you need - nothing vague.

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
- If the message raises more than one issue, address the clearest one and note
  in your answer (if responding) or handover_summary (if escalating) that other
  points were raised too - do not silently drop them.
- If what the customer describes contradicts the account record (e.g. they claim
  a plan or charge that doesn't match what's on file), do not resolve based on
  the customer's claim alone. Use ASK_FOR_INFO to clarify, or ESCALATE if the
  discrepancy itself is the issue.
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


SUPPORT_ARTICLES = load_text_file(os.path.join("data", "support_articles.md"))
ALL_ACCOUNT_RECORDS = load_text_file(os.path.join("data", "account_records.md"))


def get_account_record(customer_id: str) -> str:
    blocks = ALL_ACCOUNT_RECORDS.split("## Customer:")
    for block in blocks[1:]:
        if customer_id in block.split("\n")[0]:
            return "## Customer:" + block
    return f"No account record found for customer ID '{customer_id}'."


def call_agent(conversation: str, customer_id: str) -> str:
    account_record = get_account_record(customer_id)
    prompt = (
        f"Account record:\n{account_record}\n\n"
        f"Support articles:\n{SUPPORT_ARTICLES}\n\n"
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

    return data


app = FastAPI()


class TicketRequest(BaseModel):
    conversation: str
    customer_id: str


@app.get("/")
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