import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")
print(orders.shape)
print(orders.dtypes)
print(orders.head())
print("exact duplicates:", orders.duplicated().sum())
print("repeated order_id:", orders["order_id"].duplicated().sum())
print(orders["status"].value_counts())
print("negative delivery:", (pd.to_numeric(orders["delivery_minutes"], errors="coerce") < 0).sum())
print("bad amount:", pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum())
customers = pd.read_csv("data/raw/customers.csv")
print(customers.shape)
print(customers.isna().sum())
print(customers["city"].value_counts())
print(customers["gender"].value_counts())
import numpy as np
customers["city"] = customers["city"].str.strip().str.title()
g = customers["gender"].str.strip().str.lower().str[0]
customers["gender"] = g.map({"m": "male", "f": "female"}).fillna("unknown")
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[~customers["age"].between(10, 100), "age"] = np.nan
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median"))
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
print(customers["gender"].value_counts())
print(customers["age"].isna().sum(), customers["age"].min(), customers["age"].max())
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders = orders.drop_duplicates()
orders = orders.sort_values("order_ts").drop_duplicates("order_id", keep="last")

products = pd.read_csv("data/raw/products.csv")
orders = orders.merge(products[["product_id", "category", "price_inr"]],
                      on="product_id", how="left")

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
fill_amt = (orders["price_inr"] * orders["quantity"]
            * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(fill_amt)

orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
done = orders["status"] == "completed"
med = orders[done].groupby("category")["delivery_minutes"].transform("median")
orders.loc[done, "delivery_minutes"] = orders.loc[done, "delivery_minutes"].fillna(med)

print(len(orders), orders["status"].value_counts().to_dict())
print(orders["order_amount"].isna().sum())
print(orders.loc[done, "delivery_minutes"].isna().sum())
print(round(orders.loc[done, "order_amount"].sum(), 2))
import os, sqlite3

os.makedirs("data/clean", exist_ok=True)

orders_out = orders.drop(columns=["category", "price_inr"]).copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
customers_out = customers.copy()
customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")

customers_out.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders_out.to_csv("data/clean/orders.csv", index=False)

con = sqlite3.connect("data/instamart.db")
customers_out.to_sql("customers", con, if_exists="replace", index=False)
products.to_sql("products", con, if_exists="replace", index=False)
orders_out.to_sql("orders", con, if_exists="replace", index=False)
con.close()
print("saved clean tables")