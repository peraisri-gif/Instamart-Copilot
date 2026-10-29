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
- Oversampling (training set only, 570 per class, random_state=42): on the untouched test set, recall rose from 0.652 to 0.854, precision fell from 0.644 to 0.589, and F1 rose from 0.648 to 0.697. Recall moved the most. For the retention team this means the balanced model catches many more customers who are about to lapse, at the price of more wrongly targeted loyal customers, which only costs a small discount.
- Key coefficients: frequency -1.698 (frequent buyers rarely lapse), recency +0.040, cancel_rate -0.124 (weak).
- Retention list: top 10 Premium test customers by lapse probability, 3 of 10 truly lapsed.

### No leakage
No encoder, scaler or resampler saw test rows; no data from 2025-09-01 onwards entered the features; segment labels from task 3 were not used as features.

### Limitations and Next Steps
The Premium flag comes from whole-window RFM, which includes the September window that defines the label, so it is a post-hoc whole-window label and the retention list is illustrative, not a leakage-free evaluation.

## Part C

### Guardrails and redaction
- Input guardrails run before any model call: 4 empty tickets get `skipped_empty`, 4 tickets above MAX_INPUT_TOKENS = 400 (734 estimated tokens each) get `flagged_long`. None of them is sent to the model.
- PII redaction (email, then phone, then order id): 55 emails, 91 phones, 132 order ids, 278 substitutions in total. Re-applying the patterns finds 0 left. An assertion checks every prompt before each call.

### Retries and validation
- Every model call is wrapped in a retry function: max 3 retries, wait = 0.05 x 2^n x uniform(0.9, 1.15). A full run has 19 TransientLLMError; every one was retry number 0, so each wait is between 0.045 and 0.0575 s, and all 19 recovered. No ticket ended with a permanent error.
- Reply validation, full run over 156 eligible tickets: 135 classified_json, 12 classified_keyword (extracted with `re.findall`), 9 flagged_invalid_output (sent to human review, never guessed).

### Zero-shot vs few-shot
| Prompt | Classified | Correct | Accuracy | Total tokens | Tokens per ticket |
|---|---|---|---|---|---|
| Zero-shot | 147 | 126 | 85.7% | 8,776 | 59.7 |
| Few-shot | 147 | 147 | 100% | 21,880 | 148.8 |

Zero-shot confusion table: billing 38/38, delivery 36/36, fraud 23 correct with 11 predicted as billing, technical 29 correct with 10 predicted as billing. Every error is a fraud or technical ticket predicted as billing.

The few-shot prompt costs about 89 extra tokens per ticket (148.8 vs 59.7, about 2.5x in total) but fixes all 21 errors. Wrongly routing a fraud ticket is expensive (account takeover goes unnoticed), so the extra cost is worth paying. For cheap, low-risk categories zero-shot would be enough.

### Budget and routing
- With TOKEN_BUDGET = 6000 (zero-shot), 49 tickets end as `budget_exhausted` (outputs/triage_budget_demo.csv).
- Final run: few-shot with TOKEN_BUDGET = 100000, routing from outputs/triage_results.csv (164 rows): security_team 34 (fraud, HIGH), premium_desk 11 (premium customers, HIGH), support_queue 102 (NORMAL), human_review 17 (NORMAL).
- Rule-validated and routed automatically: 147 tickets (135 JSON + 12 keyword). Sent to humans: 17 tickets (4 empty, 4 too long, 9 invalid output).
## Part D
## Limitations and Next Steps

### Limitations
- **Premium flag is a post-hoc label.** The Premium segment comes from whole-window RFM (task 3), which includes the September window that defines the lapse label. The retention list (3 of the top 10 customers truly lapsed) is therefore illustrative and not a leakage-free evaluation. The model's own features use only orders before 2025-09-01.
- **Small test set.** The lapse model is judged on 232 customers, so test F1 (0.648) can move by several points with another split. Oversampling duplicates lapsed training rows, so it adds no new information.
- **Synthetic data and a mock model.** All data is generated, and every language-model call goes through the offline `mock_llm.py`. The 100% few-shot accuracy and the 85.7% zero-shot accuracy describe that mock, not a real model.
- **TF-IDF retrieval.** It matches exact words only, so paraphrases (for example "money back" for "refund") can be missed, and a fact split across a chunk boundary can be lost (seen with overlap 0).
- **Token counts are estimates** from `est_tokens`, not from a real tokenizer.
- **Refund agent.** It handles one order per message and treats every request as raised inside the 7-day window.

### Next Steps
- Compute the Premium flag from orders before the cutoff only, and re-evaluate the retention list without leakage.
- Use time-based validation (train on earlier months, test on later ones) and calibrate the predicted lapse probabilities.
- Replace TF-IDF with embedding-based retrieval and add a re-ranking step.
- Add more ticket categories and a confidence threshold that sends uncertain tickets to human review.
- Add unit tests for the refund rules and a human-approval step before large refunds are issued.
### Chunking and retrieval
- Chunks of 40 words with overlap 10 (stride 30): 15 chunks in total (refund 4, delivery 3, escalation 3, payment 3, privacy 2).
- Retriever: TfidfVectorizer fitted on the chunk texts, cosine similarity, top k=3. For "refund policy late delivery" all three chunks come from refund_policy.txt.
- TF measures how often a word appears in a chunk; IDF gives more weight to words that are rare across all chunks. One limitation: two phrasings with the same meaning but different words (for example "money back" and "refund") do not match, because TF-IDF only compares exact words.

### Evaluation (policy_eval.csv: 12 answerable, 2 unanswerable)
| Overlap | Chunks | Right source in top 3 | Expected fact in answer | Fallback on unanswerable |
|---|---|---|---|---|
| 10 | 15 | 12/12 | 12/12 | 2/2 |
| 0 | 13 | 12/12 | 10/12 | 2/2 |

With overlap 0 two questions fail ("What delivery time makes a completed order eligible for a full refund?" and "How soon is money returned when a payment fails but is deducted?"). The fact was split across a chunk boundary, so neither chunk held the whole sentence even though the right document was retrieved. Overlap 10 keeps the boundary words in both chunks.
The fallback answer is the one that begins "I could not find".
### Tools and ReAct agent
- Five tools (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`) are defined in `TOOLS_SPEC` and looked up through `TOOL_MAP`. `issue_refund` and `escalate_to_human` append one JSON line to `outputs/refund_ledger.jsonl` and never write the same order twice, so a resumed run cannot refund twice.
- Refund rules: only completed orders; delivery over 45 minutes gives 100% of order_amount, 45 minutes or less gives 0; a refund above INR 1,000 is escalated.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,122 |
| S2 | REFUND | 135.00 | 5 | 1,122 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,109 |
| S4 | DENIED | 0 | 4 | 785 |
| S5 | DENIED | 0 | 3 | 472 |
| S6 | NOT_FOUND | 0 | 2 | 162 |
| S7 | FALLBACK_MAX_ITERATIONS | 0 | 6 | 3,270 |

All seven match the expected outcome and refund within INR 0.01. With the default `token_limit=6000`, S7 finishes all six model calls and stops with FALLBACK_MAX_ITERATIONS.

### Safe stopping
- Max iterations: after 6 model calls without a final answer the run stops (S7).
- Token limit: with S1 and `token_limit=200` the run ended with **FALLBACK_TOKEN_LIMIT** after 3 model calls, before executing the tool that was just requested.
- Time limit: with S1 and `time_limit_seconds=0` the run ended with **FALLBACK_TIMEOUT** before the first model call (0 calls). Every fallback returns the steps completed so far and a short human-handoff message in `final_answer`.

### Resumable runs
A checkpoint (run id, scenario id, full message list, tokens used, step number, tools executed) is written to `state/<run_id>.json` after every tool observation. For S1 I raised a simulated crash right after the second tool's checkpoint, then called `resume_agent`. The final status was REFUND with INR 128.25, and the tools executed across the crash and the resume were, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund` (four executions, none repeated).

### ReAct versus plan-and-execute
WRITE YOUR OWN COMPARISON HERE (max 250 words).
My ReAct agent asks the model what to do after every observation. This costs one model call per step: S1 needed 5 calls and 1,122 tokens, S5 needed 3 calls and 472 tokens, and S6 needed only 2 calls and 162 tokens. A plan-and-execute design would ask the model for the whole step list once and then run the tools without the model, replanning only on an error. For the same refund it would need roughly 2 model calls (one for the plan, one for the final answer), so it would use fewer calls and fewer tokens than ReAct.

The difference shows up when an observation is unexpected. In S6 the order id does not exist, and in S5 the order was cancelled. ReAct reads the NOT_FOUND or the cancelled status straight away and stops after 2 or 3 calls. A fixed plan would carry on with the policy search and the refund calculation for an order that cannot be refunded, and it would only recover if a replan is triggered.

For this refund workflow I would choose ReAct. The next step depends on what each tool returns (the order status, the delivery time, and whether the amount is above INR 1,000), and a wrong step can move money. The extra cost is a few hundred tokens per ticket, which is small compared to the risk of issuing a wrong refund.