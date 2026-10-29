import os, json, time, sqlite3, glob
from datetime import datetime
import pandas as pd
import mock_llm
from rag import chunk_docs, make_retriever

os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)
LEDGER = "outputs/refund_ledger.jsonl"
RUNS = "outputs/agent_runs.jsonl"
_retrieve = make_retriever(chunk_docs(40, 10))

# ---------- Task 5: tools ----------
def _log(tool, payload):
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                            "tool": tool, **payload}) + "\n")

def _already(tool, order_id):
    if not os.path.exists(LEDGER):
        return False
    return any(json.loads(l)["tool"] == tool and json.loads(l)["order_id"] == order_id
               for l in open(LEDGER))

def get_order(order_id):
    con = sqlite3.connect("data/instamart.db")
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes "
                      "FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    d = row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": row[2],
            "delivery_minutes": None if d is None or pd.isna(d) else float(d)}

def search_policy(query):
    return {"results": _retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}

def issue_refund(order_id, refund_inr):
    if not _already("issue_refund", order_id):   # never refund an order twice
        _log("issue_refund", {"order_id": order_id, "refund_inr": refund_inr})
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    if not _already("escalate_to_human", order_id):
        _log("escalate_to_human", {"order_id": order_id, "refund_inr": refund_inr})
    return {"refund_inr": refund_inr}

TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}

def _spec(name, desc, props):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props,
                           "required": list(props)}}

TOOLS_SPEC = [
    _spec("get_order", "Look up an order in the database by its order id.",
          {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents for relevant chunks.",
          {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery time.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human lead.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]

SYSTEM = ("You are a refund AGENT for a quick-commerce company. Use the tools to look up "
          "the order, check the refund rules, then issue or escalate the refund. Finish "
          "with an answer that starts with REFUND, ESCALATE, DENIED, NEEDS_INFO or "
          "NOT_FOUND, followed by a colon.")

# ---------- Tasks 6, 7, 8: ReAct loop, limits, checkpoints ----------
class SimulatedCrash(Exception):
    pass

def _save(state):
    with open(f"state/{state['run_id']}.json", "w") as f:
        json.dump(state, f)

def _finish(state, status, final):
    state["status"], state["final_answer"] = status, final
    _save(state)
    return state

def _handoff(state, status):
    names = ", ".join(state["executed"]) or "none"
    return _finish(state, status,
                   f"HANDOFF to a human agent: stopped with {status} after "
                   f"{len(state['steps'])} steps ({names}).")

def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= time_limit_seconds:
            return _handoff(state, "FALLBACK_TIMEOUT")
        if state["llm_calls"] >= max_iterations:
            return _handoff(state, "FALLBACK_MAX_ITERATIONS")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:      # stop before running the tool
            return _handoff(state, "FALLBACK_TOKEN_LIMIT")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            return _finish(state, final.split(":")[0], final)
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        obs = TOOL_MAP[tc.name](**tc.arguments)
        state["messages"].append({"role": "tool", "tool_call_id": tc.id,
                                  "name": tc.name, "content": json.dumps(obs)})
        state["executed"].append(tc.name)
        state["steps"].append({"step": len(state["steps"]) + 1, "tool": tc.name,
                               "arguments": tc.arguments, "observation": obs})
        _save(state)                                 # checkpoint after every observation
        if crash_after_tools and len(state["executed"]) == crash_after_tools:
            raise SimulatedCrash("simulated crash after checkpoint")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "steps": [], "executed": [],
             "status": None, "final_answer": None}
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    state = json.load(open(f"state/{run_id}.json"))
    return _loop(state, max_iterations, token_limit, time_limit_seconds)

def refund_of(state):
    for s in reversed(state["steps"]):
        if s["tool"] == "calculate_refund":
            return float(s["observation"]["refund_inr"])
    return 0.0

def log_run(state):
    with open(RUNS, "a") as f:
        f.write(json.dumps({k: state[k] for k in
                ["scenario_id", "status", "final_answer", "llm_calls",
                 "tokens_used", "steps"]}) + "\n")

if __name__ == "__main__":
    for p in [LEDGER, RUNS] + glob.glob("state/*.json"):
        if os.path.exists(p):
            os.remove(p)

    sc = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
    print("id status refund calls tokens match")
    for _, r in sc.iterrows():
        st = run_agent(r["message"], r["scenario_id"])
        log_run(st)
        amt = refund_of(st) if st["status"] in ("REFUND", "ESCALATE") else 0.0
        ok_status = (st["status"] == r["expected_outcome"] or
                     (r["expected_outcome"] == "FALLBACK" and st["status"].startswith("FALLBACK")))
        ok = ok_status and abs(amt - float(r["expected_refund_inr"])) <= 0.01
        print(r["scenario_id"], st["status"], amt, st["llm_calls"], st["tokens_used"], ok)

    s1 = sc[sc["scenario_id"] == "S1"].iloc[0]["message"]
    d1 = run_agent(s1, "S1", token_limit=200, run_id="S1-token200")
    print("\ntoken demo:", d1["status"], "| calls", d1["llm_calls"])
    d2 = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1-time0")
    print("time demo:", d2["status"], "| calls", d2["llm_calls"])
    try:
        run_agent(s1, "S1", run_id="S1-crash", crash_after_tools=2)
    except SimulatedCrash:
        print("crashed after 2 tools, resuming...")
    d3 = resume_agent("S1-crash")
    print("resume:", d3["status"], refund_of(d3), "| tools:", d3["executed"])