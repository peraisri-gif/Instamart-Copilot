import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
sql = open("sql/queries.sql").read()
queries = [q.strip() for q in sql.split(";") if "SELECT" in q]

res = []
for i, q in enumerate(queries, 1):
    df = pd.read_sql_query(q, con)
    res.append(df)
    print(f"Q{i}: {len(df)} rows")

print("Q2 total:", round(res[1]["revenue"].sum(), 2), "max:", res[1].iloc[0].to_dict())
avg = pd.read_sql_query(
    "SELECT ROUND(AVG(t), 2) AS a FROM (SELECT SUM(order_amount) AS t "
    "FROM orders WHERE status='completed' GROUP BY customer_id)", con)
print("Q5 average spend:", avg["a"][0])
q6 = res[5]
print("Q6 top per city:\n", q6[q6["rn"] == 1])
print("Q6 rank != row_number:", (q6["rnk"] != q6["rn"]).sum())
print("Q7 last cumulative:", res[6]["cumulative_revenue"].iloc[-1])
con.close()