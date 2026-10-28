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