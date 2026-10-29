import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.makedirs("outputs", exist_ok=True)

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

# Task 1: RFM
snapshot = pd.Timestamp("2025-10-01 00:00:00")
rfm = done.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()
rfm["recency"] = (snapshot - rfm["last_order"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print("customers:", len(rfm))
print("recency median", rfm["recency"].median(), "min", rfm["recency"].min(), "max", rfm["recency"].max())
print("frequency mean", round(rfm["frequency"].mean(), 2), "max", rfm["frequency"].max())
print("monetary mean", round(rfm["monetary"].mean(), 2), "max", round(rfm["monetary"].max(), 2))

# Task 2: K-means for K = 2..8
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("K  inertia  silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3))
    from sklearn.decomposition import PCA

# Task 3: final K = 4
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

prof = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    recency=("recency", "mean"),
    frequency=("frequency", "mean"),
    monetary=("monetary", "mean"),
).round(2)
print(prof)

# Premium = highest mean monetary
premium_id = prof["monetary"].idxmax()
rest = prof.drop(premium_id)
names = {premium_id: "Premium High Spenders"}
names[rest["frequency"].idxmax()] = "Loyal Frequent Buyers"
names[rest["recency"].idxmax()] = "Lapsing Low Activity"
for sid in prof.index:
    names.setdefault(sid, "Occasional Buyers")
print(names)

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
rfm[["customer_id", "recency", "frequency", "monetary",
     "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)

# PCA
pca = PCA(n_components=2).fit(X)
print("PCA ratios:", pca.explained_variance_ratio_.round(3),
      "total", round(pca.explained_variance_ratio_.sum(), 3))