import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
c = pd.read_sql_query("SELECT * FROM customers", con)
o = pd.read_sql_query("SELECT * FROM orders", con)
con.close()

done = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]

checks = {
    "customers rows = 1200": len(c) == 1200,
    "city counts": c["city"].value_counts().to_dict() == {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197,
        "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts": c["gender"].value_counts().to_dict() == {
        "female": 610, "male": 560, "unknown": 30},
    "0 missing ages": c["age"].isna().sum() == 0,
    "ages within 18-60": c["age"].between(18, 60).all(),
    "orders rows = 12309": len(o) == 12309,
    "order_id unique": o["order_id"].is_unique,
    "completed = 11110": len(done) == 11110,
    "cancelled = 1199": len(canc) == 1199,
    "no pending rows": (o["status"] == "pending").sum() == 0,
    "0 missing amount": o["order_amount"].isna().sum() == 0,
    "0 missing delivery (completed)": done["delivery_minutes"].isna().sum() == 0,
    "1199 missing delivery (cancelled)": canc["delivery_minutes"].isna().sum() == 1199,
    "completed sum = 2998668.80": round(done["order_amount"].sum(), 2) == 2998668.80,
}

for name, ok in checks.items():
    print("PASS" if ok else "FAIL", "-", name)
assert all(checks.values()), "Some checkpoints failed"
print("ALL CHECKPOINTS PASSED")