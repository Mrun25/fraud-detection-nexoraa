"""
╔══════════════════════════════════════════════════════════════════╗
║  FRAUD & ANOMALY DETECTION — NEXORAA TECHNOSOLVE ASSESSMENT     ║
║  Complete Analysis Pipeline                                      ║
╚══════════════════════════════════════════════════════════════════╝

Objectives Covered:
  1. Duplicate / Fake Account Detection
  2. Suspicious Transaction Detection
  3. Bonus Abuse & Promotional Misuse
  4. Referral Fraud Detection
  5. Fraud Detection Rules (Rule Engine)
  6. Risk Scoring Methodology
  7. False Positive / False Negative Impact Analysis
  8. Monitoring Framework Design
  9. Recommendations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Styling ───────────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#1B2A4A",
    "danger":    "#E63946",
    "warning":   "#F4A261",
    "safe":      "#2A9D8F",
    "accent":    "#457B9D",
    "light":     "#F1FAEE",
    "mid":       "#A8DADC",
}
plt.rcParams.update({
    "figure.facecolor":  COLORS["light"],
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelcolor":   COLORS["primary"],
    "xtick.color":       COLORS["primary"],
    "ytick.color":       COLORS["primary"],
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})
sns.set_palette([COLORS["primary"], COLORS["danger"], COLORS["warning"],
                 COLORS["safe"], COLORS["accent"], COLORS["mid"]])

OUT = "outputs"

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
users = pd.read_csv("data/users.csv", parse_dates=["registration_date"])
txns  = pd.read_csv("data/transactions.csv", parse_dates=["transaction_date"])
refs  = pd.read_csv("data/referrals.csv",    parse_dates=["referral_date"])
bonus = pd.read_csv("data/bonuses.csv",      parse_dates=["claimed_date"])

print(f"  Users: {len(users):,}  |  Txns: {len(txns):,}  |  Referrals: {len(refs):,}  |  Bonuses: {len(bonus):,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/9] EDA...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Exploratory Data Analysis — Transaction Platform Overview",
             fontsize=16, fontweight="bold", color=COLORS["primary"], y=1.01)

# 1a Transaction amount distribution
ax = axes[0, 0]
legit = txns[~txns["_is_fraud"]]["amount"]
fraud = txns[txns["_is_fraud"]]["amount"]
ax.hist(np.log1p(legit), bins=50, alpha=0.7, color=COLORS["safe"],   label="Legitimate")
ax.hist(np.log1p(fraud), bins=50, alpha=0.7, color=COLORS["danger"], label="Fraudulent")
ax.set_title("Transaction Amount Distribution (log scale)", fontweight="bold")
ax.set_xlabel("log(Amount + 1)")
ax.set_ylabel("Count")
ax.legend()

# 1b Transactions by hour
ax = axes[0, 1]
hour_fraud  = txns[txns["_is_fraud"]]["transaction_hour"].value_counts().sort_index()
hour_legit  = txns[~txns["_is_fraud"]]["transaction_hour"].value_counts().sort_index()
ax.bar(hour_legit.index, hour_legit.values, color=COLORS["safe"],   alpha=0.7, label="Legitimate", width=0.4, align="edge")
ax.bar(hour_fraud.index - 0.4, hour_fraud.values, color=COLORS["danger"], alpha=0.7, label="Fraudulent", width=0.4, align="edge")
ax.set_title("Transaction Volume by Hour of Day", fontweight="bold")
ax.set_xlabel("Hour")
ax.set_ylabel("Count")
ax.legend()

# 1c Payment method breakdown
ax = axes[0, 2]
pm = txns["payment_method"].value_counts()
ax.barh(pm.index, pm.values, color=[COLORS["primary"], COLORS["accent"],
         COLORS["safe"], COLORS["warning"], COLORS["mid"], COLORS["danger"]])
ax.set_title("Transactions by Payment Method", fontweight="bold")
ax.set_xlabel("Count")

# 1d Transaction status pie
ax = axes[1, 0]
status = txns["status"].value_counts()
wedge_colors = [COLORS["safe"], COLORS["danger"], COLORS["warning"], COLORS["mid"]]
ax.pie(status.values, labels=status.index, autopct="%1.1f%%",
       colors=wedge_colors[:len(status)], startangle=90,
       wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Transaction Status Distribution", fontweight="bold")

# 1e Merchant category fraud rate
ax = axes[1, 1]
cat_fraud = txns.groupby("merchant_category")["_is_fraud"].mean().sort_values(ascending=True)
bars = ax.barh(cat_fraud.index, cat_fraud.values * 100,
               color=[COLORS["danger"] if v > 0.06 else COLORS["safe"] for v in cat_fraud.values])
ax.axvline(6, color=COLORS["warning"], linestyle="--", linewidth=1.5, label="Avg fraud rate")
ax.set_title("Fraud Rate by Merchant Category (%)", fontweight="bold")
ax.set_xlabel("Fraud Rate (%)")
ax.legend()

# 1f Monthly transaction volume
ax = axes[1, 2]
txns["month"] = txns["transaction_date"].dt.to_period("M")
monthly = txns.groupby(["month", "_is_fraud"]).size().unstack(fill_value=0)
monthly.index = monthly.index.astype(str)
monthly.plot(kind="bar", ax=ax, color=[COLORS["safe"], COLORS["danger"]], alpha=0.85)
ax.set_title("Monthly Transaction Volume", fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)
ax.legend(["Legitimate", "Fraudulent"])

plt.tight_layout()
plt.savefig(f"{OUT}/01_eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_eda_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DUPLICATE ACCOUNT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/9] Duplicate account detection...")

# Rule: flag accounts sharing email / phone / device_id
email_dupes  = users[users.duplicated("email",     keep=False)][["user_id","email","registration_date","country","ip_address"]]
phone_dupes  = users[users.duplicated("phone",     keep=False)][["user_id","phone","registration_date","country"]]
device_dupes = users[users.duplicated("device_id", keep=False)][["user_id","device_id","registration_date","ip_address"]]

# Suspicious: registered within 7 days AND share identifier
users["reg_ts"] = users["registration_date"].astype(np.int64) // 10**9
users_sorted = users.sort_values("email")
users["email_dup_flag"]   = users.duplicated("email",     keep=False).astype(int)
users["phone_dup_flag"]   = users.duplicated("phone",     keep=False).astype(int)
users["device_dup_flag"]  = users.duplicated("device_id", keep=False).astype(int)
users["dup_score"]        = users["email_dup_flag"] + users["phone_dup_flag"] + users["device_dup_flag"]
users["is_duplicate_flag"]= (users["dup_score"] >= 1).astype(int)

detected_dupes = users[users["is_duplicate_flag"] == 1]
precision_dupes = (detected_dupes["_is_duplicate_account"].sum() / len(detected_dupes) * 100)
recall_dupes    = (detected_dupes["_is_duplicate_account"].sum() /
                   users["_is_duplicate_account"].sum() * 100)

print(f"  Duplicate accounts detected: {len(detected_dupes):,}")
print(f"  Precision: {precision_dupes:.1f}%  |  Recall: {recall_dupes:.1f}%")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Objective 1 — Duplicate / Fake Account Detection",
             fontsize=14, fontweight="bold", color=COLORS["primary"])

ax = axes[0]
dup_counts = pd.Series({
    "Email duplicates":  email_dupes["user_id"].nunique(),
    "Phone duplicates":  phone_dupes["user_id"].nunique(),
    "Device duplicates": device_dupes["user_id"].nunique(),
})
ax.bar(dup_counts.index, dup_counts.values,
       color=[COLORS["danger"], COLORS["warning"], COLORS["accent"]])
ax.set_title("Duplicate Signals by Type", fontweight="bold")
ax.set_ylabel("Unique Users Flagged")
ax.tick_params(axis="x", rotation=15)

ax = axes[1]
score_dist = users["dup_score"].value_counts().sort_index()
ax.bar(score_dist.index, score_dist.values,
       color=[COLORS["safe"], COLORS["warning"], COLORS["danger"], COLORS["primary"]])
ax.set_title("Duplicate Signal Score Distribution", fontweight="bold")
ax.set_xlabel("Number of Matching Identifiers")
ax.set_ylabel("User Count")

ax = axes[2]
country_dupes = detected_dupes["country"].value_counts().head(7)
ax.barh(country_dupes.index, country_dupes.values, color=COLORS["danger"], alpha=0.8)
ax.set_title("Duplicate Accounts by Country", fontweight="bold")
ax.set_xlabel("Count")

plt.tight_layout()
plt.savefig(f"{OUT}/02_duplicate_accounts.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_duplicate_accounts.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SUSPICIOUS TRANSACTION DETECTION (Isolation Forest)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/9] Suspicious transaction detection...")

# Feature engineering
txns["amount_log"]   = np.log1p(txns["amount"])
txns["is_odd_hour"]  = ((txns["transaction_hour"] >= 0) & (txns["transaction_hour"] <= 5)).astype(int)
txns["is_round_amt"] = (txns["amount"] % 1000 == 0).astype(int)

# Per-user velocity: txns in last 1h window
txns_sorted = txns.sort_values(["user_id", "transaction_date"])
txns_sorted["prev_tx_time"] = txns_sorted.groupby("user_id")["transaction_date"].shift(1)
txns_sorted["time_since_last_tx_min"] = (
    (txns_sorted["transaction_date"] - txns_sorted["prev_tx_time"])
    .dt.total_seconds().fillna(9999) / 60
)
txns_sorted["rapid_tx_flag"] = (txns_sorted["time_since_last_tx_min"] < 5).astype(int)

# R12: Impossible Travel (IP velocity)
txns_sorted["prev_ip"] = txns_sorted.groupby("user_id")["ip_address"].shift(1)
txns_sorted["impossible_travel_flag"] = (
    (txns_sorted["time_since_last_tx_min"] < 120) & 
    (txns_sorted["ip_address"] != txns_sorted["prev_ip"]) & 
    (txns_sorted["prev_ip"].notna())
).astype(int)

# User-level aggregates
user_stats = txns.groupby("user_id").agg(
    tx_count          = ("transaction_id", "count"),
    avg_amount        = ("amount",          "mean"),
    max_amount        = ("amount",          "max"),
    std_amount        = ("amount",          "std"),
    international_pct = ("is_international","mean"),
    odd_hour_count    = ("is_odd_hour",     "sum"),
).fillna(0).reset_index()

# Z-score amount anomaly
user_stats["amount_zscore"] = (
    (user_stats["max_amount"] - user_stats["avg_amount"]) /
    (user_stats["std_amount"].replace(0, 1))
)

# Isolation Forest on transaction-level features
features = ["amount_log", "is_odd_hour", "is_round_amt", "rapid_tx_flag"]
X = txns_sorted[features].fillna(0)
iso = IsolationForest(n_estimators=200, contamination=0.07, random_state=42)
txns_sorted["iso_score"]  = iso.fit_predict(X)
txns_sorted["anomaly_score"] = -iso.score_samples(X)   # higher = more anomalous
txns_sorted["iso_flag"]   = (txns_sorted["iso_score"] == -1).astype(int)

# Merge back
txns = txns.merge(txns_sorted[["transaction_id", "iso_flag", "anomaly_score",
                                 "time_since_last_tx_min", "rapid_tx_flag", "impossible_travel_flag"]],
                   on="transaction_id", how="left")

# Rule-based flags
txns["high_amount_flag"]  = (txns["amount"] > txns["amount"].quantile(0.97)).astype(int)
txns["odd_hour_flag"]     = txns["is_odd_hour"]
txns["round_amount_flag"] = txns["is_round_amt"]
txns["intl_flag"]         = txns["is_international"].astype(int)

# Composite suspicion score (0–5)
txns["suspicion_score"] = (
    txns["iso_flag"] +
    txns["high_amount_flag"] +
    txns["odd_hour_flag"] +
    txns["round_amount_flag"] +
    txns["rapid_tx_flag"].fillna(0).astype(int)
)
txns["suspicious_flag"] = (txns["suspicion_score"] >= 2).astype(int)

tp = ((txns["suspicious_flag"] == 1) & (txns["_is_fraud"] == True)).sum()
fp = ((txns["suspicious_flag"] == 1) & (txns["_is_fraud"] == False)).sum()
fn = ((txns["suspicious_flag"] == 0) & (txns["_is_fraud"] == True)).sum()
tn = ((txns["suspicious_flag"] == 0) & (txns["_is_fraud"] == False)).sum()
prec = tp / (tp + fp) * 100
rec  = tp / (tp + fn) * 100
print(f"  Suspicious txns flagged: {txns['suspicious_flag'].sum():,}")
print(f"  Precision: {prec:.1f}%  |  Recall: {rec:.1f}%")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Objective 2 — Suspicious Transaction Detection",
             fontsize=14, fontweight="bold", color=COLORS["primary"])

ax = axes[0, 0]
ax.scatter(txns[~txns["_is_fraud"]]["anomaly_score"].sample(2000, random_state=42),
           txns[~txns["_is_fraud"]]["amount"].sample(2000, random_state=42),
           alpha=0.3, s=15, color=COLORS["safe"], label="Legitimate")
ax.scatter(txns[txns["_is_fraud"]]["anomaly_score"],
           txns[txns["_is_fraud"]]["amount"],
           alpha=0.7, s=20, color=COLORS["danger"], label="Fraud")
ax.set_title("Isolation Forest Anomaly Score vs Amount", fontweight="bold")
ax.set_xlabel("Anomaly Score (higher = more suspicious)")
ax.set_ylabel("Transaction Amount (INR)")
ax.legend()

ax = axes[0, 1]
score_dist = txns["suspicion_score"].value_counts().sort_index()
bar_colors = [COLORS["safe"] if i < 2 else COLORS["warning"] if i < 4 else COLORS["danger"]
              for i in score_dist.index]
ax.bar(score_dist.index, score_dist.values, color=bar_colors)
ax.axvline(1.5, color="red", linestyle="--", linewidth=1.5, label="Flag threshold")
ax.set_title("Suspicion Score Distribution", fontweight="bold")
ax.set_xlabel("Composite Suspicion Score (0–5)")
ax.set_ylabel("Transaction Count")
ax.legend()

ax = axes[1, 0]
cm = confusion_matrix(txns["_is_fraud"], txns["suspicious_flag"])
sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="RdYlGn_r",
            xticklabels=["Predicted OK", "Predicted Fraud"],
            yticklabels=["Actual OK", "Actual Fraud"])
ax.set_title(f"Confusion Matrix  (Precision={prec:.1f}%, Recall={rec:.1f}%)", fontweight="bold")

ax = axes[1, 1]
fraud_hour = txns[txns["_is_fraud"]]["transaction_hour"].value_counts().sort_index()
legit_hour = txns[~txns["_is_fraud"]]["transaction_hour"].value_counts().sort_index()
fraud_pct  = fraud_hour / (fraud_hour + legit_hour.reindex(fraud_hour.index, fill_value=1)) * 100
ax.bar(fraud_pct.index, fraud_pct.values,
       color=[COLORS["danger"] if v > 15 else COLORS["warning"] for v in fraud_pct.values])
ax.axhline(fraud_pct.mean(), color=COLORS["primary"], linestyle="--", label="Mean fraud %")
ax.set_title("Fraud % by Transaction Hour", fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Fraud %")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/03_suspicious_transactions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_suspicious_transactions.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3B — SUPERVISED ML & EXPLAINABILITY (XGBOOST + SHAP)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3B/9] Supervised ML Upgrade (XGBoost) & Explainability...")

# Train XGBoost
features_xgb = ["amount_log", "is_odd_hour", "is_round_amt", "rapid_tx_flag", "is_international", "impossible_travel_flag"]
X_xgb = txns[features_xgb].fillna(0)
y_xgb = txns["_is_fraud"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.3, random_state=42, stratify=y_xgb)
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss")
xgb_clf.fit(X_train, y_train)

# Predictions
txns["xgb_prob"] = xgb_clf.predict_proba(X_xgb)[:, 1]
txns["xgb_flag"] = (txns["xgb_prob"] > 0.5).astype(int)

# Comparison Plot
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle("Supervised vs Unsupervised ML Performance", fontsize=14, fontweight="bold", color=COLORS["primary"])

iso_score_norm = (txns["anomaly_score"] - txns["anomaly_score"].min()) / (txns["anomaly_score"].max() - txns["anomaly_score"].min())
prec_iso, rec_iso, _ = precision_recall_curve(y_xgb, iso_score_norm)
prec_xgb, rec_xgb, _ = precision_recall_curve(y_xgb, txns["xgb_prob"])

ax.plot(rec_iso, prec_iso, label=f"Isolation Forest (AUC={auc(rec_iso, prec_iso):.2f})", color=COLORS["warning"], linewidth=2)
ax.plot(rec_xgb, prec_xgb, label=f"XGBoost (AUC={auc(rec_xgb, prec_xgb):.2f})", color=COLORS["safe"], linewidth=2)
ax.set_title("Precision-Recall Curve Comparison", fontweight="bold")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/10_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 10_model_comparison.png")

# SHAP Explainability
explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_xgb)
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_xgb, show=False)
plt.title("SHAP Values: Model Explainability", fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{OUT}/11_shap_explainability.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 11_shap_explainability.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BONUS ABUSE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/9] Bonus abuse detection...")

bonus_per_user = bonus.groupby("user_id").agg(
    claim_count   = ("bonus_id",     "count"),
    total_bonus   = ("bonus_amount",  "sum"),
    unique_promos = ("promo_code",   "nunique"),
    approved      = ("status",        lambda x: (x == "approved").sum()),
).reset_index()

# Rule: >2 claims OR >1 unique promo in short window
bonus_per_user["multi_claim_flag"]  = (bonus_per_user["claim_count"]   > 2).astype(int)
bonus_per_user["multi_promo_flag"]  = (bonus_per_user["unique_promos"] > 1).astype(int)
bonus_per_user["bonus_abuse_flag"]  = ((bonus_per_user["multi_claim_flag"] == 1) |
                                        (bonus_per_user["multi_promo_flag"] == 1)).astype(int)

# Merge ground truth
bonus_per_user = bonus_per_user.merge(
    bonus.groupby("user_id")["_is_abuse"].max().reset_index(), on="user_id")

abuse_detected  = bonus_per_user[bonus_per_user["bonus_abuse_flag"] == 1]
prec_bonus = (abuse_detected["_is_abuse"].sum() / len(abuse_detected) * 100)
rec_bonus  = (abuse_detected["_is_abuse"].sum() / bonus_per_user["_is_abuse"].sum() * 100)
print(f"  Bonus abusers detected: {len(abuse_detected):,}")
print(f"  Precision: {prec_bonus:.1f}%  |  Recall: {rec_bonus:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Objective 3 — Bonus Abuse & Promotional Misuse",
             fontsize=14, fontweight="bold", color=COLORS["primary"])

ax = axes[0]
claim_dist = bonus_per_user["claim_count"].value_counts().sort_index().head(12)
bar_colors = [COLORS["danger"] if i > 2 else COLORS["safe"] for i in claim_dist.index]
ax.bar(claim_dist.index, claim_dist.values, color=bar_colors)
ax.axvline(2.5, color="red", linestyle="--", label="Abuse threshold")
ax.set_title("Bonus Claims per User Distribution", fontweight="bold")
ax.set_xlabel("Number of Claims")
ax.set_ylabel("User Count")
ax.legend()

ax = axes[1]
promo_fraud = bonus[bonus["_is_abuse"]].groupby("promo_code").size().sort_values()
ax.barh(promo_fraud.index, promo_fraud.values, color=COLORS["warning"], alpha=0.85)
ax.set_title("Abuse Incidents by Promo Code", fontweight="bold")
ax.set_xlabel("Fraudulent Claims")

ax = axes[2]
labels = ["True Positive\n(Abuse detected)", "False Positive\n(Wrongly flagged)",
          "False Negative\n(Missed abuse)", "True Negative\n(Correct clean)"]
sizes  = [
    int(abuse_detected["_is_abuse"].sum()),
    int((abuse_detected["_is_abuse"] == False).sum()),
    int(bonus_per_user["_is_abuse"].sum() - abuse_detected["_is_abuse"].sum()),
    int((bonus_per_user["bonus_abuse_flag"] == 0).sum()),
]
colors = [COLORS["safe"], COLORS["warning"], COLORS["danger"], COLORS["primary"]]
ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors,
       wedgeprops={"edgecolor": "white"})
ax.set_title(f"Detection Accuracy\n(Prec={prec_bonus:.0f}%, Rec={rec_bonus:.0f}%)", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/04_bonus_abuse.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_bonus_abuse.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — REFERRAL FRAUD DETECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/9] Referral fraud detection...")

# Self-referral: same device or same IP
refs["self_referral_flag"]   = ((refs["same_device"] == True) | (refs["same_ip"] == True)).astype(int)

# Referrer making too many referrals (ring fraud)
ref_counts = refs.groupby("referrer_id").size().reset_index(name="referral_count")
ref_counts["high_volume_flag"] = (ref_counts["referral_count"] > ref_counts["referral_count"].quantile(0.95)).astype(int)
refs = refs.merge(ref_counts[["referrer_id", "referral_count", "high_volume_flag"]], on="referrer_id")

# Referee not verified
refs["unverified_referee_flag"] = (~refs["referee_verified"]).astype(int)

refs["ref_fraud_score"] = (
    refs["self_referral_flag"] +
    refs["high_volume_flag"] +
    refs["unverified_referee_flag"]
)
refs["ref_fraud_flag"] = (refs["ref_fraud_score"] >= 2).astype(int)

prec_ref = (refs[refs["ref_fraud_flag"] == 1]["_is_fraudulent"].sum() /
            refs["ref_fraud_flag"].sum() * 100) if refs["ref_fraud_flag"].sum() > 0 else 0
rec_ref  = (refs[refs["ref_fraud_flag"] == 1]["_is_fraudulent"].sum() /
            refs["_is_fraudulent"].sum() * 100) if refs["_is_fraudulent"].sum() > 0 else 0
print(f"  Referral fraud detected: {refs['ref_fraud_flag'].sum():,}")
print(f"  Precision: {prec_ref:.1f}%  |  Recall: {rec_ref:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Objective 4 — Referral Fraud & Suspicious Referral Patterns",
             fontsize=14, fontweight="bold", color=COLORS["primary"])

ax = axes[0]
sig_counts = pd.Series({
    "Same Device":        refs["same_device"].sum(),
    "Same IP":            refs["same_ip"].sum(),
    "Unverified Referee": refs["unverified_referee_flag"].sum(),
    "High Volume Ref":    refs["high_volume_flag"].sum(),
})
ax.bar(sig_counts.index, sig_counts.values,
       color=[COLORS["danger"], COLORS["warning"], COLORS["accent"], COLORS["primary"]])
ax.set_title("Referral Fraud Signals", fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=15)

ax = axes[1]
top_referrers = ref_counts.nlargest(10, "referral_count")
ax.barh(top_referrers["referrer_id"], top_referrers["referral_count"],
        color=[COLORS["danger"] if v else COLORS["safe"]
               for v in top_referrers["high_volume_flag"]])
ax.set_title("Top 10 Referrers (Red = High Volume)", fontweight="bold")
ax.set_xlabel("Referrals Made")

ax = axes[2]
score_dist = refs["ref_fraud_score"].value_counts().sort_index()
bar_colors = [COLORS["safe"] if i < 2 else COLORS["danger"] for i in score_dist.index]
ax.bar(score_dist.index, score_dist.values, color=bar_colors)
ax.axvline(1.5, color="red", linestyle="--", label="Flag threshold")
ax.set_title("Referral Fraud Score Distribution", fontweight="bold")
ax.set_xlabel("Score")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/05_referral_fraud.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_referral_fraud.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FRAUD DETECTION RULES SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/9] Fraud rule engine summary...")

rules = {
    "R01 — Duplicate Email":         users["email_dup_flag"].sum(),
    "R02 — Duplicate Phone":         users["phone_dup_flag"].sum(),
    "R03 — Duplicate Device":        users["device_dup_flag"].sum(),
    "R04 — High Amount (>97th pct)": txns["high_amount_flag"].sum(),
    "R05 — Odd Hour (00–05)":        txns["odd_hour_flag"].sum(),
    "R06 — Round Amount (mod 1000)": txns["round_amount_flag"].sum(),
    "R07 — Rapid Velocity (<5 min)": txns["rapid_tx_flag"].fillna(0).sum(),
    "R08 — Isolation Forest Flag":   txns["iso_flag"].sum(),
    "R09 — Multi Bonus Claim (>2)":  bonus_per_user["multi_claim_flag"].sum(),
    "R10 — Self Referral (same dev/IP)": refs["self_referral_flag"].sum(),
    "R11 — High Volume Referrer":    refs["high_volume_flag"].sum(),
    "R12 — Impossible Travel":       txns["impossible_travel_flag"].fillna(0).sum(),
}

fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle("Objective 5 — Fraud Detection Rule Engine",
             fontsize=14, fontweight="bold", color=COLORS["primary"])
bar_colors = [COLORS["danger"] if v > 500 else COLORS["warning"] if v > 200 else COLORS["accent"]
              for v in rules.values()]
bars = ax.barh(list(rules.keys()), list(rules.values()), color=bar_colors)
ax.bar_label(bars, padding=3, fontsize=10)
ax.set_xlabel("Records Flagged")
ax.set_title("Records Flagged by Each Fraud Rule", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/06_fraud_rules.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 06_fraud_rules.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RISK SCORING METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/9] Risk scoring...")

# User-level risk score aggregation
user_risk = users[["user_id", "email_dup_flag", "phone_dup_flag", "device_dup_flag",
                    "is_verified", "kyc_status", "_is_duplicate_account", "_is_bonus_abuser"]].copy()

# Transaction risk signals per user
tx_risk = txns.groupby("user_id").agg(
    tx_fraud_count     = ("_is_fraud",         "sum"),
    suspicion_avg      = ("suspicion_score",    "mean"),
    iso_flags          = ("iso_flag",           "sum"),
    odd_hour_ratio     = ("is_odd_hour",        "mean"),
    intl_ratio         = ("is_international",   "mean"),
    impossible_travel  = ("impossible_travel_flag", "sum"),
).reset_index()

# Bonus risk
bon_risk = bonus_per_user[["user_id", "bonus_abuse_flag", "claim_count"]].copy()

# Referral risk
ref_risk_r = refs.groupby("referrer_id")["ref_fraud_flag"].sum().reset_index()
ref_risk_r.columns = ["user_id", "ref_fraud_as_referrer"]
ref_risk_e = refs.groupby("referee_id")["ref_fraud_flag"].sum().reset_index()
ref_risk_e.columns = ["user_id", "ref_fraud_as_referee"]

# Merge all
risk_df = user_risk.merge(tx_risk,    on="user_id", how="left")
risk_df = risk_df.merge(bon_risk,     on="user_id", how="left")
risk_df = risk_df.merge(ref_risk_r,   on="user_id", how="left")
risk_df = risk_df.merge(ref_risk_e,   on="user_id", how="left")
risk_df = risk_df.fillna(0)

# Weighted risk score (max ~100)
risk_df["risk_score"] = (
    risk_df["email_dup_flag"]       * 15 +
    risk_df["phone_dup_flag"]       * 15 +
    risk_df["device_dup_flag"]      * 10 +
    (~risk_df["is_verified"]).astype(int) * 8 +
    (risk_df["kyc_status"] == "rejected").astype(int) * 12 +
    (risk_df["kyc_status"] == "pending").astype(int)  * 5  +
    risk_df["suspicion_avg"]        * 8  +
    risk_df["iso_flags"].clip(0, 5) * 3  +
    risk_df["odd_hour_ratio"]       * 5  +
    risk_df["intl_ratio"]           * 4  +
    (risk_df["impossible_travel"] > 0).astype(int) * 15 +
    risk_df["bonus_abuse_flag"]     * 8  +
    risk_df["ref_fraud_as_referrer"].clip(0, 5) * 4 +
    risk_df["ref_fraud_as_referee"].clip(0, 5)  * 3
).clip(0, 100)

# Risk tiers
risk_df["risk_tier"] = pd.cut(risk_df["risk_score"],
                               bins=[-1, 20, 40, 65, 100],
                               labels=["Low", "Medium", "High", "Critical"])

print(f"  Risk tier distribution:")
print(risk_df["risk_tier"].value_counts().to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Objective 6 — Risk Scoring Methodology",
             fontsize=14, fontweight="bold", color=COLORS["primary"])

ax = axes[0]
ax.hist(risk_df["risk_score"], bins=40,
        color=COLORS["primary"], edgecolor="white", alpha=0.85)
for thresh, color, label in [(20, COLORS["safe"], "Low"), (40, COLORS["warning"], "Medium"),
                               (65, COLORS["danger"], "High")]:
    ax.axvline(thresh, color=color, linestyle="--", linewidth=2, label=f"{label} threshold")
ax.set_title("User Risk Score Distribution", fontweight="bold")
ax.set_xlabel("Risk Score (0–100)")
ax.set_ylabel("User Count")
ax.legend()

ax = axes[1]
tier_counts = risk_df["risk_tier"].value_counts()
colors_tier = {"Low": COLORS["safe"], "Medium": COLORS["warning"],
               "High": COLORS["danger"], "Critical": COLORS["primary"]}
ax.pie(tier_counts.values, labels=tier_counts.index, autopct="%1.1f%%",
       colors=[colors_tier[t] for t in tier_counts.index],
       wedgeprops={"edgecolor": "white"})
ax.set_title("Risk Tier Distribution", fontweight="bold")

ax = axes[2]
feat_importance = pd.Series({
    "Duplicate Email":     15,
    "Duplicate Phone":     15,
    "KYC Rejected":        12,
    "Duplicate Device":    10,
    "Bonus Abuse Flag":    8,
    "KYC Not Verified":    8,
    "Suspicion Score Avg": 8,
    "Impossible Travel":   15,
    "International Ratio": 4,
    "Referral Fraud":      4,
    "Odd Hour Ratio":      5,
})
feat_importance = feat_importance.sort_values()
ax.barh(feat_importance.index, feat_importance.values,
        color=[COLORS["danger"] if v >= 12 else COLORS["warning"] if v >= 8 else COLORS["accent"]
               for v in feat_importance.values])
ax.set_title("Risk Score Component Weights", fontweight="bold")
ax.set_xlabel("Weight (points)")

plt.tight_layout()
plt.savefig(f"{OUT}/07_risk_scoring.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 07_risk_scoring.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FALSE POSITIVE / NEGATIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/9] FP/FN analysis...")

# Simulate 3 threshold scenarios for the transaction detection
thresholds = [1, 2, 3]
results = []
for t in thresholds:
    pred = (txns["suspicion_score"] >= t).astype(int)
    actual = txns["_is_fraud"].astype(int)
    tp = ((pred == 1) & (actual == 1)).sum()
    fp = ((pred == 1) & (actual == 0)).sum()
    fn = ((pred == 0) & (actual == 1)).sum()
    tn = ((pred == 0) & (actual == 0)).sum()
    prec_t = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_t  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_t   = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0
    # Business cost: FP = lost revenue (blocked legit user ~₹500 avg), FN = fraud loss (~₹2000 avg)
    fp_cost = fp * 500
    fn_cost = fn * 2000
    results.append({
        "Threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": prec_t, "Recall": rec_t, "F1": f1_t,
        "FP_Cost_INR": fp_cost, "FN_Cost_INR": fn_cost,
        "Total_Cost_INR": fp_cost + fn_cost,
    })
results_df = pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Objective 7 — False Positive / False Negative Business Impact",
             fontsize=14, fontweight="bold", color=COLORS["primary"])

ax = axes[0, 0]
x = np.arange(len(thresholds))
w = 0.3
ax.bar(x - w, results_df["Precision"], w, label="Precision", color=COLORS["safe"])
ax.bar(x,     results_df["Recall"],    w, label="Recall",    color=COLORS["danger"])
ax.bar(x + w, results_df["F1"],        w, label="F1-Score",  color=COLORS["accent"])
ax.set_xticks(x)
ax.set_xticklabels([f"Threshold = {t}" for t in thresholds])
ax.set_ylim(0, 1.1)
ax.set_title("Precision / Recall / F1 at Different Thresholds", fontweight="bold")
ax.set_ylabel("Score")
ax.legend()

ax = axes[0, 1]
ax.bar(results_df["Threshold"].astype(str), results_df["FP_Cost_INR"] / 1e6,
       color=COLORS["warning"], label="FP Cost (blocked legit)")
ax.bar(results_df["Threshold"].astype(str), results_df["FN_Cost_INR"] / 1e6,
       bottom=results_df["FP_Cost_INR"] / 1e6,
       color=COLORS["danger"], label="FN Cost (missed fraud)")
ax.set_title("Business Cost by Threshold (₹ Millions)", fontweight="bold")
ax.set_xlabel("Suspicion Threshold")
ax.set_ylabel("Estimated Cost (₹M)")
ax.legend()

ax = axes[1, 0]
# Best threshold confusion matrix
best_thresh_row = results_df.loc[results_df["Total_Cost_INR"].idxmin()]
pred_best = (txns["suspicion_score"] >= int(best_thresh_row["Threshold"])).astype(int)
cm_best = confusion_matrix(txns["_is_fraud"].astype(int), pred_best)
sns.heatmap(cm_best, annot=True, fmt="d", ax=ax, cmap="Blues",
            xticklabels=["Predicted OK", "Predicted Fraud"],
            yticklabels=["Actual OK", "Actual Fraud"])
ax.set_title(f"Optimal Threshold = {int(best_thresh_row['Threshold'])} (Lowest Total Cost)", fontweight="bold")

ax = axes[1, 1]
categories = ["FP Impact\n(False Block)", "FN Impact\n(Fraud Loss)"]
fp_impacts = ["Lost customer trust\nReduced conversion\nSupport ticket cost\nChurn risk", ""]
fn_impacts = ["", "Direct financial loss\nRegulatory exposure\nReputational damage\nChargeback liability"]
impact_table = [
    ["Customer blocked unfairly", "Fraud transaction approved"],
    ["₹500 avg opportunity cost", "₹2,000 avg fraud loss"],
    ["High precision needed", "High recall needed"],
    ["Prefer Threshold = 3", "Prefer Threshold = 1"],
]
ax.axis("off")
table = ax.table(cellText=impact_table,
                 colLabels=["False Positive Impact", "False Negative Impact"],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(COLORS["primary"])
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 0:
        cell.set_facecolor("#FFE8E8")
    else:
        cell.set_facecolor("#E8F4FF")
ax.set_title("FP vs FN Business Impact Summary", fontweight="bold", pad=50)

plt.tight_layout()
plt.savefig(f"{OUT}/08_fp_fn_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 08_fp_fn_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MONITORING FRAMEWORK & KPIs
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9/9] Monitoring framework...")

# Simulate daily KPI monitoring
dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
np.random.seed(42)
daily_kpis = pd.DataFrame({
    "date":               dates,
    "total_txns":         np.random.randint(25, 45, len(dates)),
    "fraud_alerts":       np.random.poisson(2.5, len(dates)),
    "blocked_txns":       np.random.poisson(1.8, len(dates)),
    "new_accounts":       np.random.randint(5, 20, len(dates)),
    "duplicate_flags":    np.random.poisson(0.8, len(dates)),
    "bonus_abuse_flags":  np.random.poisson(0.5, len(dates)),
})
# Inject anomalous spikes (fraud events)
spike_days = [45, 90, 135, 220, 310]
for d in spike_days:
    daily_kpis.loc[d, "fraud_alerts"]  += np.random.randint(8, 15)
    daily_kpis.loc[d, "blocked_txns"]  += np.random.randint(5, 10)

daily_kpis["rolling_fraud_rate"] = (
    daily_kpis["fraud_alerts"].rolling(7).mean() /
    daily_kpis["total_txns"].rolling(7).mean() * 100
).fillna(0)

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
fig.suptitle("Objective 8 — Continuous Fraud Monitoring Framework & KPI Dashboard",
             fontsize=14, fontweight="bold", color=COLORS["primary"])

# KPI cards (row 0)
kpi_data = [
    ("Total Transactions", f"{len(txns):,}", COLORS["primary"]),
    ("Fraud Alerts Raised", f"{txns['suspicious_flag'].sum():,}", COLORS["danger"]),
    ("High Risk Users", f"{(risk_df['risk_tier'] == 'Critical').sum():,}", COLORS["warning"]),
    ("Duplicate Accounts", f"{users['is_duplicate_flag'].sum():,}", COLORS["accent"]),
    ("Bonus Abuse Cases", f"{bonus_per_user['bonus_abuse_flag'].sum():,}", COLORS["mid"]),
    ("Referral Fraud", f"{refs['ref_fraud_flag'].sum():,}", COLORS["safe"]),
]
for i, (label, val, color) in enumerate(kpi_data[:3]):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(color)
    ax.text(0.5, 0.6,  val,   transform=ax.transAxes, ha="center", va="center",
            fontsize=28, fontweight="bold", color="white")
    ax.text(0.5, 0.2, label,  transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="white", alpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

# Rolling fraud rate
ax = fig.add_subplot(gs[1, :2])
ax.plot(daily_kpis["date"], daily_kpis["rolling_fraud_rate"],
        color=COLORS["danger"], linewidth=2)
ax.fill_between(daily_kpis["date"], daily_kpis["rolling_fraud_rate"],
                alpha=0.15, color=COLORS["danger"])
ax.axhline(daily_kpis["rolling_fraud_rate"].mean() * 1.5, color=COLORS["warning"],
           linestyle="--", linewidth=1.5, label="Alert threshold")
for d in spike_days:
    ax.axvline(dates[d], color=COLORS["danger"], alpha=0.5, linewidth=1, linestyle=":")
ax.set_title("7-Day Rolling Fraud Rate (%)", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Fraud Rate (%)")
ax.legend()

# Risk tier over time (simulated cohort)
ax = fig.add_subplot(gs[1, 2])
tier_vals = risk_df["risk_tier"].value_counts()
colors_t  = [COLORS["safe"], COLORS["warning"], COLORS["danger"], COLORS["primary"]]
tier_order = ["Low", "Medium", "High", "Critical"]
vals_ordered = [tier_vals.get(t, 0) for t in tier_order]
ax.bar(tier_order, vals_ordered, color=colors_t)
ax.set_title("Current User Risk Tiers", fontweight="bold")
ax.set_ylabel("Users")

# Daily fraud alerts
ax = fig.add_subplot(gs[2, :2])
ax.bar(daily_kpis["date"], daily_kpis["fraud_alerts"], color=COLORS["warning"],
       width=1, alpha=0.6, label="Daily alerts")
ax.plot(daily_kpis["date"], daily_kpis["fraud_alerts"].rolling(30).mean(),
        color=COLORS["danger"], linewidth=2, label="30-day avg")
ax.set_title("Daily Fraud Alerts & 30-Day Moving Average", fontweight="bold")
ax.set_ylabel("Alert Count")
ax.legend()

# Monitoring checklist
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
checklist = [
    "✅  Real-time velocity monitoring",
    "✅  Daily duplicate account scan",
    "✅  Weekly bonus abuse audit",
    "✅  Monthly referral ring analysis",
    "✅  KYC verification enforcement",
    "✅  Automated risk score refresh",
    "✅  SLA: Review Critical alerts <2h",
    "✅  SLA: Review High alerts <24h",
    "✅  Monthly model retraining",
    "✅  Quarterly rule review",
]
for i, item in enumerate(checklist):
    color = COLORS["safe"] if "✅" in item else COLORS["danger"]
    ax.text(0.05, 0.95 - i * 0.09, item, transform=ax.transAxes,
            fontsize=9.5, va="top", color=COLORS["primary"])
ax.set_title("Monitoring Checklist", fontweight="bold", pad=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.savefig(f"{OUT}/09_monitoring_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 09_monitoring_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — EXPORT FLAGGED DATA FOR POWER BI
# ══════════════════════════════════════════════════════════════════════════════
print("\nExporting Power BI ready data...")

# Transactions enriched
txns_export = txns[[
    "transaction_id", "user_id", "transaction_date", "transaction_type",
    "amount", "currency", "payment_method", "merchant_category", "status",
    "transaction_hour", "is_international", "is_odd_hour", "is_round_amt",
    "iso_flag", "suspicion_score", "suspicious_flag", "anomaly_score",
    "_is_fraud"
]].copy()
txns_export.columns = [c.replace("_is_fraud", "ground_truth_fraud") for c in txns_export.columns]
txns_export.to_csv("data/transactions_enriched.csv", index=False)

# Users with risk scores
risk_export = risk_df.merge(
    users[["user_id", "email", "country", "registration_date", "kyc_status",
            "is_verified", "account_age_days", "total_deposits", "total_withdrawals"]],
    on="user_id")
risk_export.to_csv("data/users_risk_scored.csv", index=False)

# Daily KPIs
daily_kpis.to_csv("data/daily_kpis.csv", index=False)

# Summary stats
summary = {
    "metric": [
        "Total Users", "Total Transactions", "Total Referrals", "Total Bonus Claims",
        "Duplicate Accounts Detected", "Suspicious Transactions", "High+Critical Risk Users",
        "Bonus Abusers Detected", "Referral Fraud Detected",
        "Fraud Detection Precision (%)", "Fraud Detection Recall (%)"
    ],
    "value": [
        len(users), len(txns), len(refs), len(bonus),
        int(users["is_duplicate_flag"].sum()), int(txns["suspicious_flag"].sum()),
        int((risk_df["risk_tier"].isin(["High", "Critical"])).sum()),
        int(bonus_per_user["bonus_abuse_flag"].sum()), int(refs["ref_fraud_flag"].sum()),
        round(prec, 1), round(rec, 1)
    ]
}
pd.DataFrame(summary).to_csv("data/summary_stats.csv", index=False)

print("  Exported: transactions_enriched.csv")
print("  Exported: users_risk_scored.csv")
print("  Exported: daily_kpis.csv")
print("  Exported: summary_stats.csv")

print("\n" + "="*60)
print("✅  ALL ANALYSIS COMPLETE")
print("="*60)
print(f"  Charts saved to: {OUT}/")
print(f"  Data exports:    data/")
