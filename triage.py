import re
import pandas as pd
import mock_llm

MAX_INPUT_TOKENS = 400
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

# Task 1: input guardrails (before any model call)
def guard(text):
    if not text.strip():
        return "skipped_empty"
    if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None

tickets["guard"] = tickets["text"].apply(guard)
print(tickets["guard"].value_counts(dropna=False))
long_t = tickets[tickets["guard"] == "flagged_long"]
print("long ticket tokens:", [mock_llm.est_tokens(t) for t in long_t["text"]])

# Task 2: PII redaction, in this exact order
PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

def redact(text):
    counts = {}
    for name, pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        counts[name] = n
    return text, counts

totals = {"email": 0, "phone": 0, "order_id": 0}
redacted = []
for t in tickets["text"]:
    r, c = redact(t)
    redacted.append(r)
    for k in totals:
        totals[k] += c[k]
tickets["redacted"] = redacted
print("substitutions:", totals, "total:", sum(totals.values()))

left = sum(len(pat.findall(r)) for r in redacted for _, pat, _ in PATTERNS)
print("PII left after redaction:", left)
import json, time, random

BASE_DELAY, MAX_RETRIES = 0.05, 3
ALLOWED = ["billing", "delivery", "technical", "fraud"]

# Task 3: two prompts
ZERO_SHOT = (
    "CLASSIFY the support ticket into exactly one category: "
    "billing, delivery, technical or fraud. "
    'Reply with JSON only: {"category": "...", "reason": "..."}'
)
FEW_SHOT = ZERO_SHOT + "\nEXAMPLES:\n" + "\n".join([
    'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}',
    'Ticket: My order has not arrived and the rider is not picking up calls. -> {"category": "delivery", "reason": "late order"}',
    'Ticket: The app crashes every time I open the cart. -> {"category": "technical", "reason": "app crash"}',
    'Ticket: Someone hacked my account and used my card. -> {"category": "fraud", "reason": "unauthorized access"}',
])

# Task 4: retry with exponential backoff and jitter
def call_with_retry(messages, tid, tcount):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            tcount[0] += 1
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2**n * random.uniform(0.9, 1.15)
            print(f"RETRY ticket={tid} retry={n} wait={wait:.4f}s")
            time.sleep(wait)

# Task 5: validate the reply
def validate(reply):
    try:
        obj = json.loads(reply)
        cat = obj.get("category") if isinstance(obj, dict) else None
        if cat in ALLOWED:
            return "classified_json", cat
        return "flagged_invalid_output", ""
    except json.JSONDecodeError:
        found = {w for w in re.findall(r"[a-z]+", reply.lower()) if w in ALLOWED}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""

def run_pipeline(system_prompt, budget=None):
    mock_llm.reset()
    random.seed(42)
    tcount, total, rows = [0], 0, []
    for _, t in tickets.iterrows():
        row = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"],
               "status": "", "category": "", "tokens_used": 0,
               "true_category": t["true_category"]}
        if isinstance(t["guard"], str):
            row["status"] = t["guard"]
        elif budget is not None and total >= budget:
            row["status"] = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": t["redacted"]}]
            for m in msgs:  # no PII may reach the model
                assert not any(p.search(m["content"]) for _, p, _ in PATTERNS)
            resp = call_with_retry(msgs, t["ticket_id"], tcount)
            used = resp.usage.total_tokens
            total += used
            row["tokens_used"] = used
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), tcount[0]

def report(name, df, transients):
    cl = df[df["status"].isin(["classified_json", "classified_keyword"])]
    ok = (cl["category"] == cl["true_category"]).sum()
    print(f"\n=== {name} ===")
    print(df["status"].value_counts().to_dict())
    print("transient errors:", transients)
    print("classified:", len(cl), "correct:", ok, "accuracy:", round(ok / len(cl) * 100, 1), "%")
    print("total tokens:", df["tokens_used"].sum(),
          "per classified:", round(df["tokens_used"].sum() / len(cl), 1))
    return cl

# Task 6: compare prompts (no budget)
zs, zs_t = run_pipeline(ZERO_SHOT)
cl = report("ZERO-SHOT", zs, zs_t)
print(pd.crosstab(cl["true_category"], cl["category"]))
fs, fs_t = run_pipeline(FEW_SHOT)
report("FEW-SHOT", fs, fs_t)

# Task 7a: budget demo (zero-shot, TOKEN_BUDGET = 6000)
TOKEN_BUDGET = 6000
demo, _ = run_pipeline(ZERO_SHOT, budget=TOKEN_BUDGET)
demo.drop(columns="true_category").to_csv("outputs/triage_budget_demo.csv", index=False)
print("\nbudget", TOKEN_BUDGET, "-> budget_exhausted:", (demo["status"] == "budget_exhausted").sum())

# Task 7b: final routing (few-shot, TOKEN_BUDGET = 100000)
final, _ = run_pipeline(FEW_SHOT, budget=100000)
seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

def route(r):
    if r["status"] in HUMAN:
        return "human_review", "NORMAL"
    if r["category"] == "fraud":
        return "security_team", "HIGH"
    if r["customer_id"] in seg.index and bool(seg.loc[r["customer_id"], "is_premium"]):
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

final[["route", "priority"]] = final.apply(lambda r: pd.Series(route(r)), axis=1)
final["segment_name"] = final["customer_id"].map(
    lambda c: seg.loc[c, "segment_name"] if c in seg.index else "")
final = final[["ticket_id", "customer_id", "status", "category",
               "segment_name", "priority", "route", "tokens_used"]]
final.to_csv("outputs/triage_results.csv", index=False)
print("\ntriage_results rows:", len(final))
print(final["route"].value_counts().to_dict())