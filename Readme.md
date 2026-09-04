TRACK_ID=PS04

# What the project does

A GenAI resolution assistant for a broadband and mobile provider's support desk. Given the conversation so far, the customer's account record (plan, billing status, recent tickets), and a set of support articles, it decides one of three outcomes:

- Respond - drafts a resolution grounded in a matching support article, with a citation to that article.
- Ask for info - asks for exactly the missing information needed to proceed, when it can't otherwise resolve the request.
- Escalate - hands off to a human with a concise summary (the issue, what's established, what's already been tried) when the case is complex, uncertain, or not covered by any article.

The system never invents policy, article content, or account details that are not explicitly provided, and fails safe to escalate on any parsing or API error rather than guessing.

# How to run it

pip install -r requirements.txt
python app.py

This starts the full application at http://localhost:8000. Requires GEMINI_API_KEY set as an environment variable (see .env.example).

Example request:

curl -X POST http://localhost:8000/handle_ticket -H "Content-Type: application/json" -d '{"conversation": "My broadband installation was supposed to happen yesterday and nobody came.", "customer_id": "ACC1003"}'

# What data and documents I generated

- data/support_articles.md - synthetic support articles covering billing, wifi/broadband connection issues, mobile network issues, and plan questions.
- data/account_records.md - synthetic customer account records with varied histories: a clean record, a repeat complainer, an account with an open unresolved ticket, and a mobile-only customer.

All data was generated for this project using an LLM, as expected by the rules. No real customer data is used.

# Demo video

[link to be added]