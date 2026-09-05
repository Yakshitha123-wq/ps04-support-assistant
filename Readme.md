TRACK_ID=PS04

# What the project does

A GenAI resolution assistant for a broadband and mobile provider's support desk. Given the conversation so far, the customer's account record (plan, billing status, recent tickets), and a set of support articles, it decides one of three outcomes:

- Respond - drafts a resolution grounded in the most relevant support article (found via retrieval, not by dumping every article into the prompt), with a citation to that article.
- Ask for info - asks for exactly the missing information needed to proceed, when it can't otherwise resolve the request.
- Escalate - hands off to a human with a concise summary (the issue, what's established, what's already been tried) when the case is complex, uncertain, or not covered by any article.

Retrieval works by embedding each support article and the incoming customer message with gemini-embedding-001, then selecting the top 3 most relevant articles by cosine similarity (computed locally with numpy) before calling gemini-3.5-flash-lite to generate the final decision.

The system never invents policy, article content, or account details that are not explicitly provided, and fails safe to escalate on any parsing or API error rather than guessing.

A simple web interface (served from app.py) lets you pick a test customer, type a message, and see the color-coded decision returned.

# How to run it

pip install -r requirements.txt
python app.py

This starts the full application at http://localhost:8000. Requires GEMINI_API_KEY set as an environment variable (see .env.example). On first run, it computes and caches article embeddings to data/article_embeddings.json - subsequent runs load this cache instantly instead of recomputing.

Example request:

curl -X POST http://localhost:8000/handle_ticket -H "Content-Type: application/json" -d '{"conversation": "My broadband installation was supposed to happen yesterday and nobody came.", "customer_id": "ACC1003"}'

# What data and documents I generated

- data/support_articles.md - synthetic support articles covering billing, wifi/broadband connection issues, mobile network issues, and plan questions.
- data/account_records.md - synthetic customer account records with varied histories: a clean record, a repeat complainer, an account with an open unresolved ticket, a mobile-only customer, and a broadband-only customer.
- data/article_embeddings.json - precomputed embeddings for the support articles, generated with gemini-embedding-001, committed so startup doesn't need live embedding calls.

All data was generated for this project using an LLM, as expected by the rules. No real customer data is used.

# Testing

All three decision branches, the null case, and several edge cases were tested against the live Gemini API (gemini-3.5-flash-lite). Each result was manually verified against the underlying account/article data, not just checked for a plausible-looking answer.

| # | Scenario | Account | Expected | Result | Time (s) |
|---|---|---|---|---|---|
| 1 | Installation delay, open ticket exists | ACC1003 | escalate | ✅ Escalated, cited ticket #4550 | 4.02 |
| 2 | Routine slow internet | ACC1001 | respond | ✅ Responded, cited Article C2 | 1.61 |
| 3 | Thank-you message (null case) | ACC1001 | respond, no citation | ✅ Responded, citation: null | 2.28 |
| 4 | Repeat billing complaint | ACC1002 | escalate | ✅ Escalated, cited tickets #4471 and #4602 | 2.14-2.46 |
| 5 | Ambiguous SIM issue | ACC1004 | ask_for_info | ✅ Asked to clarify, did not guess | 2.04 |
| 6 | Mobile data limit | ACC1004 | respond | ✅ Responded, cited Article P2, correct plan limit | 4.55 |
| 7 | Nonexistent customer ID | ACC9999 | escalate | ✅ Escalated, explained no record found | 2.04 |
| 8 | Empty conversation | any | escalate (server-side) | ✅ Escalated, caught before any API call | 0.02 |
| 9 | Two-issue message, both resolvable | ACC1001 | respond, both cited | ✅ Responded, cited both Article P1 and B1 | 2.46-4.83 |
| 10 | Three-issue message with a contradiction | ACC1001 | escalate | ✅ Escalated, correctly flagged the billing contradiction | 2.88-5.44 |
| 11 | Multi-issue message and repeat ticket history | ACC1002 | escalate | ✅ Escalated, combined both signals correctly | 4.58 |
| 12 | Broadband-only account | ACC1005 | respond or ask_for_info | ✅ Asked for clarification per Article C2's own instructions | 2.04 |

All requests completed well within the 60-second per-request limit. The slowest real case (a 3-issue message triggering 3 embedding calls) took under 6 seconds, and the empty-conversation fail-safe returns in under 0.03 seconds since it never reaches the API.

# Fresh-clone verification

As instructed, the repository was cloned into a completely separate folder and run with only the two required commands (pip install -r requirements.txt, then python app.py), with a fresh .env containing only the API key. The app started in 1.4 seconds - confirming the precomputed embeddings cache loads correctly rather than recomputing 12 embeddings live. Three spot-check requests (a routine case, an escalation, and the two-issue citation case) were re-run against this fresh clone and matched the results from the original development copy exactly, including timing (2.3-4.1 seconds per request).

# A known limit

Retrieval caps at 5 total articles per request (2 per detected issue segment, deduplicated). This comfortably covers realistic support messages (1-4 issues), but a message with many more distinct issues than that could exceed the cap and leave one issue without grounding material. This was a deliberate tradeoff to keep prompt size and latency predictable rather than unbounded.

# A judgment call worth noting

Multi-issue messages (e.g. "I want to upgrade my plan, also is my last bill correct?") initially caused the system to escalate both issues, even when both were fully answerable from the account data, and only cite one of the two articles actually used. Two fixes were made: the system prompt now only escalates a multi-issue message when at least one issue genuinely requires human judgment, and it lists every article used in the citation field rather than just one. Retrieval itself was also changed from embedding the whole message as a single vector to splitting it into segments and retrieving per segment, so that each issue in a compound message gets a fair chance at surfacing its own relevant article. This reflects the "the hard cases, the edge cases" judgement criterion: recognizing when a routine-looking complexity is not actually complex, and making sure the grounding mechanism keeps up with that judgement.

# Three more bugs found by testing the same message across every account

Running each preset scenario against all five test accounts (not just the one it was designed for) surfaced three real, reproducible gaps that a narrower test pass would have missed:

1. Decision mislabeling: for a small number of accounts, a clarifying question was returned as decision "respond" instead of "ask_for_info", even though the content was a question, not a resolution. Fixed at two layers: the system prompt now explicitly forbids putting a question inside an "answer" field, and the code independently reclassifies any "respond" whose answer ends in a question mark, so the bug is caught even if the model ignores the instruction.
2. Unverified claims stated as fact: when an account record had no data related to the customer's claim (e.g. a customer without any recorded broadband service claiming an installation was missed), the "established" field sometimes restated the claim as if it were confirmed rather than flagging it as unverified, and one case (a customer whose broadband was already installed 6 months ago) missed a genuine contradiction entirely. Fixed by requiring the "established" field to explicitly distinguish "customer claims X" from "record confirms X", and to state contradictions explicitly rather than passing them through silently.
3. Service-type mismatch: a mobile-only account was asked a standard wifi troubleshooting question without the system noting that the account has no broadband service at all. Fixed by requiring an explicit check of whether the account record actually has the type of service being asked about.

All three fixes were re-verified by rerunning the full 12-scenario suite plus the specific account/message combinations that originally exposed each bug.

# Demo video

[link to be added]