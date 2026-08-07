"""
Fraud Detection Dataset Generator
Nexoraa Technosolve IT Services - Assessment
Generates synthetic but realistic payment platform data with embedded fraud patterns.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import hashlib
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_USERS         = 2000
N_TRANSACTIONS  = 10000
N_REFERRALS     = 600
START_DATE      = datetime(2024, 1, 1)
END_DATE        = datetime(2024, 12, 31)

# Fraud injection rates
DUPLICATE_ACCOUNT_RATE  = 0.04   # 4%  of users have a duplicate
FRAUD_TX_RATE           = 0.06   # 6%  of transactions are suspicious
BONUS_ABUSE_RATE        = 0.05   # 5%  of users abused bonus
REFERRAL_FRAUD_RATE     = 0.08   # 8%  of referrals are fraudulent


def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))


def device_id():
    return "DEV-" + hashlib.md5(str(random.random()).encode()).hexdigest()[:8].upper()


def ip_address(region="normal"):
    if region == "vpn":
        # Known VPN/proxy subnets
        return f"185.{random.randint(100,200)}.{random.randint(0,255)}.{random.randint(1,254)}"
    return fake.ipv4_private()


# ══════════════════════════════════════════════════════════════════════════════
# 1. USERS TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("Generating users...")

users = []
email_pool = [fake.email() for _ in range(N_USERS)]
phone_pool = [fake.phone_number() for _ in range(N_USERS)]

for i in range(N_USERS):
    uid = f"USR{str(i+1).zfill(5)}"
    reg_date = random_date(START_DATE, END_DATE - timedelta(days=30))
    is_duplicate = random.random() < DUPLICATE_ACCOUNT_RATE
    is_bonus_abuser = random.random() < BONUS_ABUSE_RATE

    user = {
        "user_id":          uid,
        "username":         fake.user_name(),
        "email":            email_pool[i],
        "phone":            phone_pool[i],
        "registration_date": reg_date,
        "country":          random.choices(
                                ["India", "USA", "UK", "Germany", "Nigeria", "Brazil", "Russia"],
                                weights=[40, 25, 10, 8, 7, 5, 5])[0],
        "device_id":        device_id(),
        "ip_address":       ip_address("vpn" if is_duplicate else "normal"),
        "account_age_days": (END_DATE - reg_date).days,
        "is_verified":      random.choices([True, False], weights=[85, 15])[0],
        "kyc_status":       random.choices(["verified", "pending", "rejected"], weights=[70, 20, 10])[0],
        "referral_code":    f"REF{str(i+1).zfill(5)}",
        "referred_by":      None,
        "bonus_claimed":    is_bonus_abuser,
        "total_deposits":   0.0,
        "total_withdrawals": 0.0,
        # fraud labels (ground truth for evaluation)
        "_is_duplicate_account": is_duplicate,
        "_is_bonus_abuser":      is_bonus_abuser,
    }
    users.append(user)

# Inject duplicate accounts: share email/phone/device with existing user
n_dupes = int(N_USERS * DUPLICATE_ACCOUNT_RATE)
dupe_source_indices = random.sample(range(N_USERS), n_dupes)

for idx in dupe_source_indices:
    users[idx]["_is_duplicate_account"] = True
    # Find a different user and copy their identifiers
    target_idx = random.choice([j for j in range(N_USERS) if j != idx])
    dupe_type = random.choice(["email", "phone", "device_id"])
    if dupe_type == "email":
        users[idx]["email"] = users[target_idx]["email"]
    elif dupe_type == "phone":
        users[idx]["phone"] = users[target_idx]["phone"]
    else:
        users[idx]["device_id"] = users[target_idx]["device_id"]

df_users = pd.DataFrame(users)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRANSACTIONS TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("Generating transactions...")

TRANSACTION_TYPES   = ["deposit", "withdrawal", "transfer", "payment", "refund"]
PAYMENT_METHODS     = ["credit_card", "debit_card", "upi", "net_banking", "wallet", "crypto"]
MERCHANT_CATEGORIES = ["gaming", "retail", "food", "travel", "utilities", "entertainment", "crypto_exchange"]

transactions = []
user_ids = df_users["user_id"].tolist()

for i in range(N_TRANSACTIONS):
    tid = f"TXN{str(i+1).zfill(6)}"
    uid = random.choice(user_ids)
    tx_date = random_date(START_DATE, END_DATE)
    tx_type = random.choices(TRANSACTION_TYPES, weights=[35, 25, 20, 15, 5])[0]

    is_fraud = random.random() < FRAUD_TX_RATE
    fraud_type = None

    if is_fraud:
        fraud_type = random.choice(["velocity", "large_amount", "odd_hour", "multiple_cards", "round_amount"])
        if fraud_type == "large_amount":
            amount = round(random.uniform(5000, 50000), 2)
        elif fraud_type == "round_amount":
            amount = float(random.choice([1000, 5000, 10000, 25000, 50000]))
        else:
            amount = round(random.uniform(100, 3000), 2)
        hour = random.choice([1, 2, 3, 4])   # odd hours
        tx_date = tx_date.replace(hour=hour)
    else:
        amount = round(np.random.lognormal(mean=5.5, sigma=1.2), 2)  # realistic log-normal spend
        amount = max(1.0, min(amount, 4999.0))
        hour = random.randint(7, 22)
        tx_date = tx_date.replace(hour=hour)

    transaction = {
        "transaction_id":   tid,
        "user_id":          uid,
        "transaction_date": tx_date,
        "transaction_type": tx_type,
        "amount":           amount,
        "currency":         random.choices(["INR", "USD", "EUR", "GBP"], weights=[55, 25, 12, 8])[0],
        "payment_method":   random.choice(PAYMENT_METHODS),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "merchant_id":      f"MERCH{str(random.randint(1,200)).zfill(4)}",
        "status":           random.choices(["success", "failed", "pending", "reversed"],
                                           weights=[80, 8, 7, 5])[0],
        "device_id":        df_users.loc[df_users["user_id"] == uid, "device_id"].values[0],
        "ip_address":       ip_address("vpn" if is_fraud else "normal"),
        "transaction_hour": tx_date.hour,
        "is_international": random.random() < 0.15,
        # ground truth
        "_is_fraud":        is_fraud,
        "_fraud_type":      fraud_type,
    }
    transactions.append(transaction)

df_transactions = pd.DataFrame(transactions)

# Inject velocity fraud: same user, many transactions in short window
velocity_users = random.sample(user_ids, 30)
base_velocity_txns = []
for uid in velocity_users:
    burst_time = random_date(START_DATE, END_DATE - timedelta(hours=2))
    for j in range(random.randint(8, 15)):
        tid = f"TXN_V{str(len(transactions)+len(base_velocity_txns)+1).zfill(6)}"
        base_velocity_txns.append({
            "transaction_id":   tid,
            "user_id":          uid,
            "transaction_date": burst_time + timedelta(minutes=j*4),
            "transaction_type": "payment",
            "amount":           round(random.uniform(50, 500), 2),
            "currency":         "INR",
            "payment_method":   "credit_card",
            "merchant_category": "gaming",
            "merchant_id":      f"MERCH{str(random.randint(1,200)).zfill(4)}",
            "status":           "success",
            "device_id":        df_users.loc[df_users["user_id"] == uid, "device_id"].values[0],
            "ip_address":       ip_address("vpn"),
            "transaction_hour": burst_time.hour,
            "is_international": False,
            "_is_fraud":        True,
            "_fraud_type":      "velocity",
        })

df_velocity = pd.DataFrame(base_velocity_txns)
df_transactions = pd.concat([df_transactions, df_velocity], ignore_index=True)
df_transactions = df_transactions.sort_values("transaction_date").reset_index(drop=True)

# Update user totals
for uid in user_ids:
    user_txns = df_transactions[df_transactions["user_id"] == uid]
    dep = user_txns[user_txns["transaction_type"] == "deposit"]["amount"].sum()
    wdw = user_txns[user_txns["transaction_type"] == "withdrawal"]["amount"].sum()
    df_users.loc[df_users["user_id"] == uid, "total_deposits"] = round(dep, 2)
    df_users.loc[df_users["user_id"] == uid, "total_withdrawals"] = round(wdw, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 3. REFERRALS TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("Generating referrals...")

referrals = []
for i in range(N_REFERRALS):
    referrer = random.choice(user_ids)
    referee  = random.choice([u for u in user_ids if u != referrer])
    is_ref_fraud = random.random() < REFERRAL_FRAUD_RATE

    referral = {
        "referral_id":      f"REF{str(i+1).zfill(5)}",
        "referrer_id":      referrer,
        "referee_id":       referee,
        "referral_date":    random_date(START_DATE, END_DATE),
        "bonus_amount":     random.choice([50, 100, 150, 200]),
        "bonus_status":     random.choices(["paid", "pending", "rejected"], weights=[65, 25, 10])[0],
        "referee_verified": random.choices([True, False], weights=[75, 25])[0],
        "same_device":      is_ref_fraud and random.random() < 0.7,
        "same_ip":          is_ref_fraud and random.random() < 0.6,
        "_is_fraudulent":   is_ref_fraud,
    }
    referrals.append(referral)

df_referrals = pd.DataFrame(referrals)

# Update referred_by in users
for _, row in df_referrals.iterrows():
    df_users.loc[df_users["user_id"] == row["referee_id"], "referred_by"] = row["referrer_id"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. BONUS / PROMOTIONS TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("Generating bonuses...")

promo_codes = ["WELCOME50", "SUMMER100", "CASHBACK20", "FESTIVE200", "NEWUSER75"]
bonuses = []

for i, uid in enumerate(random.sample(user_ids, min(800, N_USERS))):
    is_abuse = df_users.loc[df_users["user_id"] == uid, "_is_bonus_abuser"].values[0]
    n_claims = random.randint(3, 8) if is_abuse else random.randint(1, 2)

    for j in range(n_claims):
        bonuses.append({
            "bonus_id":     f"BON{str(len(bonuses)+1).zfill(5)}",
            "user_id":      uid,
            "promo_code":   random.choice(promo_codes),
            "claimed_date": random_date(START_DATE, END_DATE),
            "bonus_amount": random.choice([50, 75, 100, 200]),
            "status":       random.choices(["approved", "rejected", "clawback"], weights=[70, 20, 10])[0],
            "_is_abuse":    is_abuse,
        })

df_bonuses = pd.DataFrame(bonuses)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE ALL
# ══════════════════════════════════════════════════════════════════════════════
out = "/home/claude/fraud-detection/data"
df_users.to_csv(f"{out}/users.csv", index=False)
df_transactions.to_csv(f"{out}/transactions.csv", index=False)
df_referrals.to_csv(f"{out}/referrals.csv", index=False)
df_bonuses.to_csv(f"{out}/bonuses.csv", index=False)

print(f"\n✅ Dataset generated successfully!")
print(f"   Users:        {len(df_users):,}")
print(f"   Transactions: {len(df_transactions):,}")
print(f"   Referrals:    {len(df_referrals):,}")
print(f"   Bonuses:      {len(df_bonuses):,}")
print(f"\n   Fraud stats:")
print(f"   Duplicate accounts: {df_users['_is_duplicate_account'].sum()}")
print(f"   Fraudulent txns:    {df_transactions['_is_fraud'].sum()}")
print(f"   Fraudulent refs:    {df_referrals['_is_fraudulent'].sum()}")
print(f"   Bonus abuse cases:  {df_bonuses['_is_abuse'].sum()}")
