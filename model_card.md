# Model Card — D2C Customer Churn Predictor

This model card details the training, performance, limitations, and ethical considerations of the D2C Customer Churn prediction model.

---

## 1. Model Details
* **Developed By:** Data Science Team
* **Model Date:** June 13, 2026
* **Model Type:** Extreme Gradient Boosting (XGBoost) Classifier
* **Version:** 1.0.0
* **Artifact:** `model.pkl` (serialized XGBoost pipeline with data preprocessor)
* **License:** Proprietary (Internal CRM tool use only)

---

## 2. Intended Use
* **Primary Intended Use:** Identify customers of a direct-to-consumer (D2C) brand who are likely to churn (i.e. place zero orders) in the next 60 days.
* **Target Users:** CRM Operations, Customer Support Teams, and Marketing Campaign Managers.
* **Out-of-Scope Uses:** Do not use this model to automatically deny service or flag profiles for fraudulent behavior. It is designed solely for positive retention incentives.

---

## 3. Factors & Features
The model utilizes 25 features spanning:
* **Demographics:** `city_tier`, `age_group`.
* **Marketing:** `acquisition_channel`, `marketing_consent`, `days_since_signup`.
* **RFM Behavior (180 Days):** `recency_days`, `frequency_180d`, `monetary_180d`, `return_rate_180d`, `avg_discount_pct_180d`.
* **Support Ticket logs (90 Days):** `ticket_count_90d`, `negative_ticket_rate_90d`, `avg_resolution_hours_90d`.
* **Web/App logs (30 Days):** `sessions_30d`, `product_views_30d`, `cart_adds_30d`, `wishlist_adds_30d`, `abandoned_carts_30d`, `last_visit_days_ago`.

---

## 4. Quantitative Metrics
Evaluation was performed on a pre-assigned, clean test split (336 customers) that was held out from training and validation.

### A. Performance on Test Set (Threshold = 0.50)
* **Accuracy:** 81.25%
* **Precision:** 81.07%
* **Recall:** 81.55%
* **F1-Score:** 81.31%
* **ROC-AUC:** 86.93%

### B. Business Optimization Performance (Threshold = 0.10)
To minimize the business cost of false negatives (missing actual churners), the optimal threshold was set to **0.10**:
* **Validation Recall:** 95.92% (captures almost all churners)
* **Validation Precision:** 56.85%
* **Cost Reduction:** Reduces expected retention costs from **49,800 INR** (at 0.50) to **23,550 INR** (at 0.10).

---

## 5. Limitations & Caveats
* **Abrupt Churn (Silent Churn):** The model cannot easily predict sudden churn from highly active, VIP customers who stop purchasing due to external factors (e.g. competitor switching) without raising complaints.
* **Feature Scope:** The model assumes the snapshot date is exactly `2025-09-30`. Feature distributions must align with this reference frame.
* **Web Data:** Web logs are restricted to a 30-day window. Longer-term app usage patterns are not captured.

---

## 6. Ethical Considerations & Fairness
* **Fairness Audit:** Ensure that retention rewards (discounts) are not distributed in a biased manner based on sensitive attributes (such as `age_group` or `city_tier`).
* **Privacy:** Customer IDs are anonymized. No personally identifiable information (PII) like names, emails, or phone numbers are exposed to the model.
* **Consent:** The model respects the `marketing_consent` flag. Customers who have opted out should not be targetted with promotional retention emails.
