import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
orders = orders.merge(prod[["product_id", "category"]], on="product_id")

cutoff = pd.Timestamp("2025-09-01")
end = pd.Timestamp("2025-10-01")

# Only orders BEFORE the cutoff feed the features
before = orders[orders["order_ts"] < cutoff]
done_b = before[before["status"] == "completed"]

feat = done_b.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
    fav_category=("category", lambda s: s.mode().iloc[0]),
).reset_index()
feat["recency"] = (cutoff - feat["last_order"]).dt.days
feat = feat.drop(columns="last_order")

canc = before.groupby("customer_id")["status"].apply(
    lambda s: (s == "cancelled").mean()).rename("cancel_rate").reset_index()
feat = feat.merge(canc, on="customer_id")

feat = feat.merge(cust[["customer_id", "city", "age", "gender", "signup_date"]],
                  on="customer_id")
feat["tenure_days"] = (cutoff - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")

# Label: no completed order from 2025-09-01 to 2025-09-30
window = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] < end)
                & (orders["status"] == "completed")]
active = set(window["customer_id"])
feat["lapsed"] = (~feat["customer_id"].isin(active)).astype(int)

print("cohort:", len(feat))
print("lapsed:", feat["lapsed"].sum(), round(feat["lapsed"].mean() * 100, 2), "%")

y = feat["lapsed"]
X = feat.drop(columns=["lapsed"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("train:", len(X_train), "test:", len(X_test))

base = [0] * len(y_test)
print("baseline accuracy:", round(accuracy_score(y_test, base), 3),
      "F1:", f1_score(y_test, base, zero_division=0))
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Task 5: preprocessing, fitted on training rows only
X_train = X_train.drop(columns="customer_id")
X_test_ids = X_test["customer_id"].copy()
X_test = X_test.drop(columns="customer_id")

cat_cols = ["city", "gender", "fav_category"]
num_cols = ["recency", "frequency", "monetary", "avg_discount",
            "avg_delivery", "cancel_rate", "age", "tenure_days"]

def make_prep():
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

prep = make_prep().fit(X_train)
scaler = prep.named_transformers_["num"]
print("scaler mean of frequency (train only):", round(scaler.mean_[1], 3))
print("full cohort mean of frequency:", round(X["frequency"].mean(), 3))
print("any missing in features:", X.drop(columns="customer_id").isna().any().any())

# Task 6: tuning (preprocessing is refit inside every CV fold)
lr_pipe = Pipeline([("prep", make_prep()),
                    ("clf", LogisticRegression(max_iter=1000))])
lr_grid = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]},
                       cv=5, scoring="f1").fit(X_train, y_train)

rf_pipe = Pipeline([("prep", make_prep()),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_grid = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8],
                                 "clf__min_samples_leaf": [5, 10, 20]},
                       cv=5, scoring="f1").fit(X_train, y_train)

for name, g in [("LogReg", lr_grid), ("RandomForest", rf_grid)]:
    pred = g.predict(X_test)
    print(name, "best params:", g.best_params_, "CV F1:", round(g.best_score_, 3))
    print(confusion_matrix(y_test, pred))
    print("precision", round(precision_score(y_test, pred), 3),
          "recall", round(recall_score(y_test, pred), 3),
          "test F1", round(f1_score(y_test, pred), 3))
    import joblib

selected = lr_grid.best_estimator_
rf_best = rf_grid.best_estimator_

# Task 7: oversample the TRAINING set only
train = X_train.copy()
train["lapsed"] = y_train
maj = train[train["lapsed"] == 0]
mino = train[train["lapsed"] == 1]
extra = mino.sample(n=len(maj) - len(mino), replace=True, random_state=42)
bal = pd.concat([train, extra])
print("balanced counts:", bal["lapsed"].value_counts().to_dict())

lr_bal = Pipeline([("prep", make_prep()),
                   ("clf", LogisticRegression(
                       max_iter=1000, C=lr_grid.best_params_["clf__C"]))]
                  ).fit(bal.drop(columns="lapsed"), bal["lapsed"])

for name, m in [("original", selected), ("balanced", lr_bal)]:
    p = m.predict(X_test)
    print(name, "precision", round(precision_score(y_test, p), 3),
          "recall", round(recall_score(y_test, p), 3),
          "F1", round(f1_score(y_test, p), 3))

# Task 8a: feature importance and coefficient signs
names = rf_best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names)
print("Top 5 RF features:\n", imp.sort_values(ascending=False).head(5).round(3))

lr_names = selected.named_steps["prep"].get_feature_names_out()
coef = pd.Series(selected.named_steps["clf"].coef_[0], index=lr_names)
print("LR coefficients:\n", coef[["num__recency", "num__frequency", "num__cancel_rate"]].round(3))

# Task 8b: save, reload, compare
os.makedirs("outputs", exist_ok=True)
joblib.dump(selected, "outputs/lapse_model.joblib")
loaded = joblib.load("outputs/lapse_model.joblib")
print("reloaded predictions identical:",
      (loaded.predict(X_test) == selected.predict(X_test)).all())

# Task 8c: retention list (Premium customers in the test set)
seg = pd.read_csv("outputs/customer_segments.csv")
prem_ids = set(seg.loc[seg["is_premium"], "customer_id"])
res = pd.DataFrame({
    "customer_id": X_test_ids.values,
    "lapse_probability": selected.predict_proba(X_test)[:, 1].round(4),
    "actually_lapsed": y_test.values,
})
res = res[res["customer_id"].isin(prem_ids)]
res = res.sort_values("lapse_probability", ascending=False).head(10)
res.to_csv("outputs/retention_list.csv", index=False)
print(res)
print("truly lapsed in top list:", res["actually_lapsed"].sum(), "of", len(res))