# Instamart Customer Intelligence & Support Copilot

## Overview
(Part A)-Synthetic-data capstone: clean order exports , segment customers and
predict lapse (Part B), triage support tickets (Part C), policy RAG assistant
and refund agent (Part D). All data is fictional; all money is in INR.

## Reproduce
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python data_pipeline.py`
6. `python validate.py`
7. `python run_queries.py`
8. `python analysis.py`

## Part A

### Audit of raw data
- orders.csv: 13,084 rows, 119 exact duplicates, 656 pending, 12,309 distinct order_id
- 120 non-numeric order_amount, 106 negative delivery_minutes, 2,541 blank delivery_minutes
- customers.csv: 1,200 rows, 61 blank ages, 6 ages outside 10-100, 24 raw city spellings, 10 raw gender spellings, 30 blank genders

### Cleaning rules (customers)
1. City: stripped spaces and applied title case, leaving exactly 6 cities (24 spellings reduced to 6).
2. Gender: mapped all spellings to male / female / unknown (blank became unknown). Result: female 610, male 560, unknown 30.
3. Age: values outside 10-100 set to missing (6 values), then all missing ages (61 blank + 6) filled with the median age of the customer's city. Result: 0 missing, range 18-60.
4. signup_date converted to datetime.

### Cleaning rules (orders), in order
1. order_ts converted to datetime.
2. Exact duplicates removed: 119 rows.
3. Repeated order_id: kept the latest order_ts only (removes the earlier pending records). Rows after this: 12,309.
4. order_amount converted to numeric (120 unparseable became missing), then filled with round(price_inr x quantity x (1 - discount_pct/100), 2). 0 missing afterwards.
5. Negative delivery_minutes (106) set to missing. Missing delivery for completed orders filled with the median delivery time of the order's product category (groupby + transform). Cancelled orders left missing (1,199) because they were never delivered.

### Result
12,309 orders: 11,110 completed, 1,199 cancelled, 0 pending. Sum of completed order_amount: INR 2,998,668.80.

### SQL (sql/queries.sql)
- Q1: 30 customers (WHERE filters rows before grouping, HAVING filters groups after).
- Q2: 30 rows, total INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45.
- Q3: 11 products. Q4: 25 customers. Q5: 550 customers, average INR 2,552.06.
- Q6: RANK differs from ROW_NUMBER in 8 rows because tied totals share a RANK but get different ROW_NUMBERs.
- Q7: 183 daily rows, final cumulative INR 2,998,668.80.

### pandas analysis
- Pivot table grand total equals the SQL total (INR 2,998,668.80).
- Per city, Bengaluru sum 523,643.40, mean 276.47, count 1,894; Mumbai sum 479,058.50, mean 297.00, count 1,613.
- Mean 269.91 vs median 135.00, skew 4.61 (right-skewed). Q1 64.80, Q3 297.00, upper fence 645.30: 897 outliers above, none below.
- Largest correlation: order_amount and discount_pct at -0.095; order_amount and quantity +0.087. Correlation does not establish causation.

### Four questions (EDA answer)
This data is about quick-commerce orders from 1,200 customers across six Indian cities between April and September 2025. Looking only at the 11,110 completed orders, total revenue comes to INR 2,998,668.80.

The order amounts are heavily skewed to the right. The median order is just INR 135, but the mean is INR 269.91, because a small number of expensive orders drag the average up. The skewness of 4.61 confirms this, and the middle half of orders sits between INR 64.80 and INR 297.

There are plenty of outliers: 897 orders are above the upper fence of INR 645.30, and none are below the lower one. I think these are real purchases of pricier items like electronics rather than mistakes, so I kept them.

The relationships are all weak. The strongest is order amount against discount percentage at -0.095, then order amount against quantity at +0.087. Delivery time barely relates to anything. Even the strongest correlation is small, and correlation alone does not prove one thing causes the other.


## Part B

### RFM and K-means
- RFM table: 1,175 customers; snapshot 2025-10-01; recency median 19 (0-180), frequency mean 9.46 (max 33), monetary mean INR 2,552.06 (max INR 14,766.55).
- K-means (StandardScaler, n_init=10, random_state=42), inertia / silhouette: K=2 1820.8 / 0.465, K=3 1268.2 / 0.445, K=4 743.1 / 0.526, K=5 617.4 / 0.481, K=6 503.2 / 0.482, K=7 408.5 / 0.416, K=8 366.5 / 0.412.
- I chose K=4: it has the highest silhouette, the elbow flattens after it, and four groups are easy for the business to act on. Inertia always falls as K grows (K=8 is lower than K=2), so inertia alone cannot pick K.

### Segments
| Name | Size | Recency | Frequency | Monetary | Action |
|---|---|---|---|---|---|
| Premium High Spenders | 138 | 24.12 | 6.77 | 7,129.70 | early access, premium perks |
| Loyal Frequent Buyers | 419 | 8.77 | 19.40 | 3,410.02 | loyalty rewards |
| Occasional Buyers | 436 | 28.86 | 3.77 | 1,057.62 | bundle offers |
| Lapsing Low Activity | 182 | 109.04 | 2.20 | 685.99 | win-back discount |

Segment ids are only labels with no ordering. PCA (2 components) explains 0.634 + 0.218 = 0.852 of the variance.

### Lapse prediction
- Cutoff 2025-09-01. Cohort 1,156 customers, 443 lapsed (38.32%). Train 924, test 232. Baseline "never lapses": accuracy 61.6%, F1 0.
- Logistic regression (best C=100): CV F1 0.651, test F1 0.648, recall 0.652. Random forest (depth 8, leaf 10): CV F1 0.693, test F1 0.633.
- **Selected: logistic regression.** Missing a lapsing customer costs a lost customer, while wrongly targeting a loyal one only costs a small discount, so recall matters most. It has the higher test F1 and recall and is easy to explain.
- - Oversampling (training set only, 570 per class, random_state=42): on the untouched test set, recall rose from 0.652 to 0.854, precision fell from 0.644 to 0.589, and F1 rose from 0.648 to 0.697. Recall moved the most. For the retention team this means the balanced model catches many more customers who are about to lapse, at the price of more wrongly targeted loyal customers, which only costs a small discount.
- Key coefficients: frequency -1.698 (frequent buyers rarely lapse), recency +0.040, cancel_rate -0.124 (weak).
- Retention list: top 10 Premium test customers by lapse probability, 3 of 10 truly lapsed.

### No leakage
No encoder, scaler or resampler saw test rows; no data from 2025-09-01 onwards entered the features; segment labels from task 3 were not used as features.

### Limitations and Next Steps
The Premium flag comes from whole-window RFM, which includes the September window that defines the label, so it is a post-hoc whole-window label and the retention list is illustrative, not a leakage-free evaluation.