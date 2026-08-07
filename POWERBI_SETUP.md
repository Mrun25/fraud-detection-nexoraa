# Power BI Dashboard Setup Guide
## Fraud & Anomaly Detection — Nexoraa Assessment

---

## Files to Import

| File | Purpose |
|------|---------|
| `data/transactions_enriched.csv` | Primary fact table — all transactions with fraud flags |
| `data/users_risk_scored.csv` | User dimension with risk scores and tiers |
| `data/daily_kpis.csv` | Time-series for trend charts |
| `data/referrals.csv` | Referral fraud signals |
| `data/bonuses.csv` | Bonus abuse records |

---

## Step 1 — Load Data

1. Open **Power BI Desktop** → Get Data → Text/CSV
2. Import all 5 CSV files listed above
3. In **Transform Data**, verify:
   - `transaction_date` → Date/Time type
   - `amount` → Decimal Number
   - `risk_score` → Decimal Number
   - `suspicious_flag`, `iso_flag` → Whole Number

---

## Step 2 — Create Relationships

In the **Model view**, connect:
- `transactions_enriched[user_id]` → `users_risk_scored[user_id]` (Many-to-One)
- `referrals[referrer_id]` → `users_risk_scored[user_id]` (Many-to-One)
- `bonuses[user_id]` → `users_risk_scored[user_id]` (Many-to-One)

---

## Step 3 — DAX Measures

Create these measures (right-click table → New Measure):

```dax
Total Transactions = COUNTROWS(transactions_enriched)

Fraud Alert Count = SUM(transactions_enriched[suspicious_flag])

Fraud Rate % = 
DIVIDE(
    SUM(transactions_enriched[suspicious_flag]),
    COUNTROWS(transactions_enriched),
    0
) * 100

High Risk Users = 
CALCULATE(
    COUNTROWS(users_risk_scored),
    users_risk_scored[risk_tier] IN {"High", "Critical"}
)

Avg Risk Score = AVERAGE(users_risk_scored[risk_score])

Total Fraud Volume = 
CALCULATE(
    SUM(transactions_enriched[amount]),
    transactions_enriched[suspicious_flag] = 1
)

Rolling 7D Fraud Rate = 
AVERAGEX(
    DATESINPERIOD(daily_kpis[date], LASTDATE(daily_kpis[date]), -7, DAY),
    DIVIDE([Fraud Alert Count], [Total Transactions], 0) * 100
)
```

---

## Step 4 — Dashboard Pages

### Page 1: Executive Overview
| Visual | Type | Fields |
|--------|------|--------|
| KPI Card | Card | Total Transactions |
| KPI Card | Card | Fraud Alert Count |
| KPI Card | Card | High Risk Users |
| KPI Card | Card | Fraud Rate % |
| Fraud by Category | Bar Chart | merchant_category vs suspicious_flag |
| Risk Tier Donut | Donut Chart | risk_tier count |
| Fraud Rate Trend | Line Chart | daily_kpis[date] vs rolling_fraud_rate |

### Page 2: Transaction Deep Dive
| Visual | Type | Fields |
|--------|------|--------|
| Amount Distribution | Histogram | amount (bins) |
| Transactions by Hour | Column Chart | transaction_hour vs count |
| Payment Method Table | Table | payment_method, count, fraud_rate |
| Suspicion Score Dist | Column Chart | suspicion_score vs count |
| Scatter: Amount vs Anomaly | Scatter | anomaly_score vs amount, colored by suspicious_flag |

### Page 3: User Risk Analysis
| Visual | Type | Fields |
|--------|------|--------|
| Risk Score Distribution | Histogram | risk_score |
| Top Risky Users | Table | user_id, risk_score, risk_tier, kyc_status |
| Risk by Country | Filled Map | country vs avg risk_score |
| KYC Status Breakdown | Donut | kyc_status vs count |
| Duplicate Signal Counts | Bar | dup_score components |

### Page 4: Bonus & Referral Fraud
| Visual | Type | Fields |
|--------|------|--------|
| Bonus Claims per User | Histogram | claim_count |
| Top Promo Codes Abused | Bar Chart | promo_code vs abuse count |
| Referral Network Table | Table | referrer_id, referral_count, suspicious_referrals |
| Fraud Signal Cards | Cards | same_device count, same_ip count |

---

## Step 5 — Formatting Tips

- Theme: Dark or "Executive" built-in theme
- Color fraud elements **Red (#E63946)**
- Color safe/legit elements **Teal (#2A9D8F)**
- Add **Fraud Risk** slicer (risk_tier) to all pages
- Add **Date Range** slicer using transaction_date

---

## Step 6 — Export Screenshots

After building:
1. File → Export → Export to PDF (for README)
2. Or: Snip each page with Windows Snipping Tool
3. Save as: `powerbi_page1_overview.png` etc.
4. Place in repo `/dashboard_screenshots/` folder
