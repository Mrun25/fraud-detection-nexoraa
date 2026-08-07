# 🔍 Fraud & Anomaly Detection System
### Nexoraa Technosolve IT Services — Data Analyst Technical Assessment

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![SQL](https://img.shields.io/badge/SQL-SQLite%2FPostgreSQL-orange?logo=sqlite)](sql/fraud_detection_queries.sql)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi)](POWERBI_SETUP.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

An online payment platform has identified unusual transaction patterns and suspects fraudulent activities. This project delivers a **complete end-to-end fraud detection framework** covering:

- Synthetic dataset generation (10,322 transactions, 2,000 users)
- Exploratory data analysis
- 11 deterministic fraud detection rules
- ML-based anomaly detection (Isolation Forest)
- Weighted risk scoring methodology (0–100 scale)
- Business impact analysis of false positives and negatives
- A continuous monitoring framework with KPI dashboards
- Power BI dashboard (5 pages) + interactive HTML dashboard

---

## 📊 Dashboard Screenshots

### Executive Overview
![Executive Overview](Executive%20overview.png)

### Transaction Analysis
![Transaction Analysis](Transaction%20Analysis.png)

### User Risk Analysis
![User Risk Analysis](User%20Risk%20Analysis.png)

### Bonus and Referral Fraud
![Bonus and Referral Fraud](Bonus%20and%20Referral%20Fraud.png)

**Interactive HTML Dashboard:** Open `dashboard.html` in any browser (no server required)

Charts included:
| Chart | Description |
|-------|-------------|
| `01_eda_overview.png` | Platform overview: amounts, hourly patterns, merchant breakdown |
| `02_duplicate_accounts.png` | Duplicate signal types, score distribution, country breakdown |
| `03_suspicious_transactions.png` | Isolation Forest scatter, suspicion score distribution, confusion matrix |
| `04_bonus_abuse.png` | Claim frequency, promo code abuse, detection accuracy |
| `05_referral_fraud.png` | Fraud signals, top referrers, score distribution |
| `06_fraud_rules.png` | All 11 rules and their flagged record counts |
| `07_risk_scoring.png` | Risk score histogram, tier distribution, component weights |
| `08_fp_fn_analysis.png` | Threshold trade-off, business cost analysis |
| `09_monitoring_dashboard.png` | KPI cards, rolling fraud rate, daily alert volume |

---

## 📁 Repository Structure

```
fraud-detection/
│
├── generate_dataset.py          # Synthetic data generator
├── fraud_detection_analysis.py  # Main analysis pipeline (9 objectives)
├── dashboard.html               # Interactive HTML dashboard
├── requirements.txt
├── POWERBI_SETUP.md             # Step-by-step Power BI guide
│
├── data/
│   ├── users.csv                      # 2,000 users with demographics
│   ├── transactions.csv               # 10,322 transactions (raw)
│   ├── referrals.csv                  # 600 referral records
│   ├── bonuses.csv                    # 1,334 bonus claims
│   ├── transactions_enriched.csv      # Transactions + all fraud flags
│   ├── users_risk_scored.csv          # Users + risk scores + tiers
│   ├── daily_kpis.csv                 # Daily KPI time-series (365 days)
│   └── summary_stats.csv             # High-level summary metrics
│
├── sql/
│   └── fraud_detection_queries.sql   # 11 rules + composite risk score query
│
└── outputs/
    ├── 01_eda_overview.png
    ├── 02_duplicate_accounts.png
    ├── 03_suspicious_transactions.png
    ├── 04_bonus_abuse.png
    ├── 05_referral_fraud.png
    ├── 06_fraud_rules.png
    ├── 07_risk_scoring.png
    ├── 08_fp_fn_analysis.png
    └── 09_monitoring_dashboard.png
```

---

## ⚙️ Setup & Execution

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/fraud-detection-nexoraa.git
cd fraud-detection-nexoraa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the synthetic dataset
python generate_dataset.py

# 4. Run the full analysis (generates all 9 charts + exports)
python fraud_detection_analysis.py

# 5. Open the interactive dashboard
# Double-click dashboard.html in your file browser
# OR: python -m http.server 8080  →  http://localhost:8080/dashboard.html
```

---

## 🗄️ Dataset Description

The dataset simulates a realistic payment platform with **injected fraud patterns** at known rates:

| Table | Records | Description |
|-------|---------|-------------|
| `users` | 2,000 | User profiles, KYC status, device/IP fingerprints |
| `transactions` | 10,322 | Payments, withdrawals, transfers with timestamps |
| `referrals` | 600 | Referral links, bonus status, device/IP overlap flags |
| `bonuses` | 1,334 | Promo code claims with abuse indicators |

**Injected fraud rates:**

| Fraud Type | Rate | Count |
|-----------|------|-------|
| Duplicate accounts | 4% | ~159 accounts |
| Fraudulent transactions | 6% + velocity bursts | ~929 txns |
| Bonus abuse | 5% of users | ~100 users |
| Referral fraud | 8% of referrals | ~41 referrals |

---

## 🔬 Analysis Methodology

### Objective 1 — Duplicate / Fake Account Detection

**Method:** Exact-match deduplication on email, phone, and device_id fields.

```python
users["email_dup_flag"]  = users.duplicated("email",     keep=False).astype(int)  # +15 pts
users["phone_dup_flag"]  = users.duplicated("phone",     keep=False).astype(int)  # +15 pts
users["device_dup_flag"] = users.duplicated("device_id", keep=False).astype(int)  # +10 pts
```

**Result:** 165 accounts flagged · Precision 50.3% · Recall 52.2%

---

### Objective 2 — Suspicious Transaction Detection

**Method:** Two-layer detection combining a rule engine + Isolation Forest (unsupervised ML).

The Isolation Forest was chosen because:
- No labeled fraud data required for training
- O(n) prediction time at inference
- Handles multi-dimensional anomalies simultaneously

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(n_estimators=200, contamination=0.07, random_state=42)
txns["iso_flag"] = (iso.fit_predict(X) == -1).astype(int)
```

**Composite suspicion score (0–5):** Each of 5 rule signals adds 1 point. Threshold ≥ 2 = flagged.

**Result:** 642 flagged · **Precision 98.9%** · Recall 68.4%

---

### Objective 3 — Bonus Abuse Detection

**Rules:**
- More than 2 claims for the same promo code → `BONUS_MULTI_CLAIM`
- Claims for more than 2 unique promo codes → `BONUS_FARMING`

**Result:** 338 users flagged · Recall 100% (very permissive rule, high FP rate expected)

---

### Objective 4 — Referral Fraud Detection

**Signals detected:**
1. Referrer and referee share the same `device_id` → `SELF_REFERRAL`
2. Referrer and referee share the same IP address → `SELF_REFERRAL`
3. Referee account unverified within 30 days → `UNVERIFIED_REFEREE`
4. Referrer in top 5th percentile of volume → `HIGH_VOLUME_REFERRER`

**Result:** 17 referrals flagged · Precision 70.6% · Recall 29.3%

---

### Objective 5 — Fraud Detection Rules

| Rule ID | Name | Signal | Weight |
|---------|------|--------|--------|
| R01 | Duplicate Email | Shared email across accounts | +15 |
| R02 | Duplicate Phone | Shared phone number | +15 |
| R03 | Device Sharing | Same device_id, multiple accounts | +10 |
| R04 | High-Value Anomaly | Amount > 97th percentile | +8 |
| R05 | Odd-Hour Transaction | Hour 00:00–05:59 | +8 |
| R06 | Round Amount | Amount mod 1000 = 0, ≥ 1000 | +5 |
| R07 | Velocity Burst | >5 transactions in 60 minutes | +8 |
| R08 | Isolation Forest | ML anomaly score flag | +5 |
| R09 | Bonus Multi-Claim | >2 claims same promo code | +8 |
| R10 | Self-Referral | Same device/IP, referrer & referee | +10 |
| R11 | High-Volume Referrer | Top 5th percentile referral count | +6 |

---

### Objective 6 — Risk Scoring Methodology

Each user receives a **composite risk score (0–100)** by aggregating weighted rule signals across all tables.

| Risk Tier | Score Range | Count | Action |
|-----------|-------------|-------|--------|
| 🟢 Low | 0–20 | 1,776 (88.8%) | No action |
| 🟡 Medium | 21–40 | 218 (10.9%) | Weekly audit |
| 🔴 High | 41–65 | 6 (0.3%) | Review within 24h |
| ⛔ Critical | 66–100 | 0 (0%) | Auto-block + 2h review |

---

### Objective 7 — False Positive / False Negative Impact

The **threshold selection problem** is the central trade-off in any fraud detection system:

| Threshold | Precision | Recall | F1 | FP Cost (₹) | FN Cost (₹) | Total Cost (₹) |
|-----------|-----------|--------|----|-------------|-------------|----------------|
| 1 | 31.4% | 100% | 47.8% | 1,430,000 | 0 | 1,430,000 |
| **2** | **98.9%** | **68.4%** | **81.1%** | **3,000** | **585,200** | **588,200** |
| 3 | 99.8% | 15.2% | 26.3% | 600 | 1,575,200 | 1,575,800 |

**Optimal strategy:** Threshold = 2 minimises total business cost.

> **False Positives** (legitimate users wrongly blocked): ~₹500 lost revenue per blocked transaction + churn risk  
> **False Negatives** (missed fraud): ~₹2,000 direct loss per fraudulent transaction + regulatory/chargeback liability

---

### Objective 8 — Monitoring Framework

**Real-time monitoring (per transaction):**
- Velocity check: flag if >5 txns in 60 min
- Suspicion score computed and stored per transaction

**Daily automated jobs:**
- Duplicate account scan across all new registrations
- Bonus abuse audit for claims in past 24h
- KPI report: fraud rate, alert count, new high-risk users

**Weekly reviews:**
- Medium-risk user batch review
- Referral ring analysis
- Rule performance audit (precision/recall tracking)

**Monthly tasks:**
- Fraud rule weight recalibration
- Model retraining (Isolation Forest with new data)
- Cross-team report: fraud, product, compliance

**SLA commitments:**
- Critical risk alerts: Manual review within **2 hours**
- High risk alerts: Review within **24 hours**
- Medium risk: Weekly batch
- Model retraining: **Monthly**

---

### Objective 9 — Recommendations

| Priority | Recommendation | Expected Impact |
|----------|----------------|----------------|
| 🔴 Critical | KYC mandatory before first withdrawal | Eliminates ~60% of account fraud exit paths |
| 🔴 Critical | Real-time velocity throttling (API rate limit) | Prevents card-testing and burst fraud |
| 🔴 Critical | Promo codes tied to verified identity (not account) | Eliminates bonus farming |
| 🟠 High | Device fingerprinting at registration | Blocks duplicate account creation |
| 🟠 High | Geo-velocity anomaly detection | Catches impossible travel/VPN fraud |
| 🟠 High | Referral bonus with 30-day verified transaction window | Eliminates inactive account referral rings |
| 🟢 Medium | Upgrade to supervised XGBoost/LightGBM | +15% recall improvement with labeled data |
| 🟢 Medium | Automated Slack/email alerts on fraud rate spike | Faster incident response |
| 🟢 Medium | Tiered review SLAs by risk score | Prioritises analyst time correctly |

---

## 💡 Key Business Insights

1. **Gaming merchant category** has a fraud rate of **18.7%** — 3× the platform average. Targeted rules recommended for this category.

2. **Transactions between midnight and 5 AM** account for only 6.6% of volume but **68% of fraud alerts**.

3. **Top 11 users** by risk score account for >35% of all bonus claims — a small cohort driving disproportionate abuse.

4. **Velocity fraud** is the most common pattern: 30 users generating 330+ burst transactions in tight time windows.

5. **Threshold = 2** minimises total business cost at ₹5.88L — compared to ₹14.3L at threshold = 1 (too aggressive) and ₹15.76L at threshold = 3 (too permissive).

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| ML / Anomaly Detection | Scikit-learn (Isolation Forest) |
| Visualisation | Matplotlib, Seaborn, Chart.js (HTML) |
| Database | SQL (SQLite compatible) |
| BI Dashboard | Power BI Desktop |
| Version Control | Git, GitHub |
| Data Generation | Faker |

---

## 👤 Author

**Mrunmayee** · B.E. Artificial Intelligence & Data Science  
Zeal College of Engineering and Research, Pune (SPPU) · CGPA 9.71

*Submitted for: Nexoraa Technosolve IT Services Pvt. Ltd. — Data Analyst Position*

---

*Dataset is entirely synthetic. All user data, transactions, and patterns are computer-generated for assessment purposes only.*
