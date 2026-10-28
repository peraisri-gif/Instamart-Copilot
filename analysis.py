import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
o = pd.read_sql_query("SELECT * FROM orders WHERE status='completed'", con)
c = pd.read_sql_query("SELECT customer_id, city FROM customers", con)
p = pd.read_sql_query("SELECT product_id, category FROM products", con)
con.close()
d = o.merge(c, on="customer_id").merge(p, on="product_id")

# (a) pivot table, margins=True
pt = pd.pivot_table(d, index="city", columns="category",
                    values="order_amount", aggfunc="sum", margins=True)
print(pt.round(2))
print("pivot grand total:", round(pt.loc["All", "All"], 2))

rep = d.groupby("city")["order_amount"].agg(
    total="sum", average="mean", orders="count").round(2)
print(rep)

# (b) distribution + outliers
x = d["order_amount"]
q1, q3 = x.quantile(0.25), x.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("mean", round(x.mean(), 2), "median", x.median(), "skew", round(x.skew(), 2))
print("Q1", q1, "Q3", q3, "lower fence", lo, "upper fence", hi)
print("above fence", (x > hi).sum(), "below fence", (x < lo).sum())

corr = d[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr().round(3)
print(corr)
print(d[x > hi]["category"].value_counts())