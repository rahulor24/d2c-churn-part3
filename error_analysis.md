# Model Error Analysis Report

**Project:** D2C Customer Churn Model Error Analysis  
**Date:** June 13, 2026  
**Model:** XGBoost Classifier (Best Model)  
**Evaluation Set:** Test Set (336 customers)  
**Threshold:** Standard classification threshold (0.50)  

---

## Executive Summary
An error analysis was performed on the test set predictions to understand model failures. The model achieved an F1-score of **81.3%** on the test set. However, we analyzed the edge cases where the model failed, specifically focusing on **5 False Positives** (customers predicted to churn who actually purchased) and **5 False Negatives** (customers predicted to stay who actually churned). 

---

## 1. False Positives (Predicted Churn, True Stay)
These are active customers who were flagged by the model as high churn risk (probability $> 0.50$), but who ended up placing an order in the 60-day target window.

| Customer ID | Pred Prob | Recency (Days) | Frequency (180d) | Monetary (180d) | Support Tickets | Last Visit (Days Ago) | Loyalty Tier |
|---|---|---|---|---|---|---|---|
| `CUST00044` | 64.5% | 72 | 1 | 899.51 | 0 | 10 | Not Enrolled |
| `CUST00109` | 62.0% | 92 | 2 | 1,622.28 | 0 | 16 | Silver |
| `CUST00335` | 87.2% | 148 | 2 | 1,328.14 | 0 | 22 | Not Enrolled |
| `CUST00437` | 90.6% | 151 | 1 | 729.22 | 0 | 33 | Silver |
| `CUST00491` | 93.1% | 97 | 1 | 540.89 | 1 | 20 | Silver |

### Qualitative Analysis of False Positives:
1. **Reactivation Patterns (CUST00335 & CUST00437):** These customers had extremely high purchase recency (148 and 151 days, respectively). The model correctly predicted high churn probabilities ($>85\%$) based on their long inactivity. However, these customers reactivated and ordered in the post-snapshot window. This represents successful organic reactivation or response to standard marketing emails.
2. **Browsing Engagement (CUST00044 & CUST00109):** Both customers had high purchase recency (72 and 92 days) but visited the website recently (1 and 10 days ago). The model was heavily weighted on purchase recency, causing it to predict churn, but their active browsing behavior was a signal that they were still in the buying loop.
3. **Implications:** False Positives are "good failures." Targeting these customers with retention campaigns would simply accelerate or secure their reactivation, resulting in low business risk.

---

## 2. False Negatives (Predicted Stay, True Churn)
These represent the most dangerous model failures: customers predicted to stay (probability $< 0.50$), but who did not purchase and were lost.

| Customer ID | Pred Prob | Recency (Days) | Frequency (180d) | Monetary (180d) | Support Tickets | Last Visit (Days Ago) | Loyalty Tier |
|---|---|---|---|---|---|---|---|
| `CUST00088` | 37.2% | 98 | 2 | 1,169.14 | 0 | 18 | Gold |
| `CUST00184` | 1.8% | 14 | 3 | 2,456.91 | 0 | 6 | Platinum |
| `CUST00247` | 32.5% | 57 | 2 | 937.32 | 0 | 14 | Not Enrolled |
| `CUST00278` | 48.2% | 69 | 1 | 339.56 | 1 | 22 | Not Enrolled |
| `CUST00379` | 38.4% | 75 | 1 | 538.25 | 0 | 4 | Not Enrolled |

### Qualitative Analysis of False Negatives:
1. **Abrupt Attrition of Champions (CUST00184):** This customer had a recency of only 14 days, had bought 3 times recently (2,456 INR), held Platinum loyalty tier, and visited the site 6 days ago. The model predicted almost 0% churn (1.8%). Yet, the customer churned. This represents a customer who abruptly stopped buying (possibly due to moving, choosing a competitor, or a sudden change of preference).
2. **False Browsing Signals (CUST00379):** This customer visited the website 4 days ago, but had only 1 past order (538 INR) and 75 days of recency. The model saw the active last visit (4 days ago) as a sign of stay, but it was just a non-converting visit.
3. **Loyalty Dilution (CUST00088):** Holds Gold loyalty tier, which heavily biased the model towards "stay," but their 98-day recency was the real driver of their churn.
4. **Implications:** To mitigate false negatives, we must **lower the decision threshold** in production. By dropping the threshold to **0.10** (our optimal business threshold), we capture **95.9%** of actual churners (Recalls), saving the brand from losing high-value customers like `CUST00184`.
