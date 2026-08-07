-- ╔══════════════════════════════════════════════════════════════════════╗
-- ║  FRAUD & ANOMALY DETECTION — SQL RULE ENGINE                       ║
-- ║  Nexoraa Technosolve IT Services — Technical Assessment            ║
-- ║  Compatible with: SQLite, PostgreSQL, MySQL, DuckDB                ║
-- ╚══════════════════════════════════════════════════════════════════════╝


-- ════════════════════════════════════════════════════════════════════════
-- RULE R01 — Duplicate Email Accounts
-- Detects multiple accounts registered with the same email address
-- ════════════════════════════════════════════════════════════════════════
SELECT
    u1.user_id          AS account_1,
    u2.user_id          AS account_2,
    u1.email            AS shared_email,
    u1.registration_date AS reg_date_1,
    u2.registration_date AS reg_date_2,
    ABS(JULIANDAY(u2.registration_date) - JULIANDAY(u1.registration_date)) AS days_apart,
    'DUPLICATE_EMAIL' AS fraud_rule
FROM users u1
JOIN users u2
    ON  u1.email   = u2.email
    AND u1.user_id < u2.user_id
ORDER BY days_apart ASC;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R02 — Duplicate Phone Accounts
-- ════════════════════════════════════════════════════════════════════════
SELECT
    u1.user_id AS account_1,
    u2.user_id AS account_2,
    u1.phone   AS shared_phone,
    u1.country AS country_1,
    u2.country AS country_2,
    'DUPLICATE_PHONE' AS fraud_rule
FROM users u1
JOIN users u2
    ON  u1.phone   = u2.phone
    AND u1.user_id < u2.user_id;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R03 — Duplicate Device Accounts
-- Same device_id used to register multiple accounts (highest risk)
-- ════════════════════════════════════════════════════════════════════════
SELECT
    device_id,
    COUNT(DISTINCT user_id)              AS account_count,
    GROUP_CONCAT(user_id, ', ')          AS user_ids,
    MIN(registration_date)               AS first_seen,
    MAX(registration_date)               AS last_seen,
    COUNT(DISTINCT country)              AS distinct_countries,
    'DEVICE_SHARING' AS fraud_rule
FROM users
GROUP BY device_id
HAVING COUNT(DISTINCT user_id) > 1
ORDER BY account_count DESC;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R04 — High-Value Transaction Anomaly
-- Transactions > 97th percentile of amount
-- ════════════════════════════════════════════════════════════════════════
WITH percentiles AS (
    SELECT
        CAST(amount AS FLOAT) AS amount,
        NTILE(100) OVER (ORDER BY amount) AS pct
    FROM transactions
),
threshold AS (
    SELECT MAX(amount) AS p97_threshold
    FROM percentiles
    WHERE pct <= 97
)
SELECT
    t.transaction_id,
    t.user_id,
    t.transaction_date,
    t.amount,
    t.payment_method,
    t.merchant_category,
    t.status,
    th.p97_threshold,
    ROUND(t.amount / th.p97_threshold, 2) AS amount_ratio,
    'HIGH_VALUE_ANOMALY' AS fraud_rule
FROM transactions t
CROSS JOIN threshold th
WHERE t.amount > th.p97_threshold
ORDER BY t.amount DESC;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R05 — Odd-Hour Transactions (00:00 – 05:59)
-- Transactions outside normal business hours
-- ════════════════════════════════════════════════════════════════════════
SELECT
    t.transaction_id,
    t.user_id,
    t.transaction_date,
    t.transaction_hour,
    t.amount,
    t.payment_method,
    t.transaction_type,
    u.country,
    u.kyc_status,
    'ODD_HOUR_TRANSACTION' AS fraud_rule
FROM transactions t
JOIN users u ON t.user_id = u.user_id
WHERE t.transaction_hour BETWEEN 0 AND 5
ORDER BY t.transaction_date DESC;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R06 — Round Amount Transactions
-- Exact multiples of 1000 are statistically rare in organic spend
-- ════════════════════════════════════════════════════════════════════════
SELECT
    transaction_id,
    user_id,
    transaction_date,
    amount,
    payment_method,
    merchant_category,
    status,
    'ROUND_AMOUNT' AS fraud_rule
FROM transactions
WHERE amount % 1000 = 0
  AND amount >= 1000
ORDER BY amount DESC;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R07 — Velocity Fraud (Multiple transactions in short window)
-- Flags users with >5 transactions within any 60-minute window
-- ════════════════════════════════════════════════════════════════════════
SELECT
    t1.user_id,
    t1.transaction_date                              AS window_start,
    DATETIME(t1.transaction_date, '+60 minutes')     AS window_end,
    COUNT(t2.transaction_id)                         AS tx_count_in_window,
    SUM(t2.amount)                                   AS total_amount_in_window,
    GROUP_CONCAT(t2.transaction_id, ', ')            AS transaction_ids,
    'VELOCITY_FRAUD' AS fraud_rule
FROM transactions t1
JOIN transactions t2
    ON  t2.user_id         = t1.user_id
    AND t2.transaction_date BETWEEN t1.transaction_date
                                AND DATETIME(t1.transaction_date, '+60 minutes')
    AND t2.transaction_id != t1.transaction_id
GROUP BY t1.user_id, t1.transaction_date
HAVING COUNT(t2.transaction_id) >= 5
ORDER BY tx_count_in_window DESC;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R08 — Bonus Multi-Claim Abuse
-- Users claiming the same promo code more than once
-- ════════════════════════════════════════════════════════════════════════
SELECT
    user_id,
    promo_code,
    COUNT(bonus_id)          AS total_claims,
    SUM(bonus_amount)        AS total_bonus_received,
    MIN(claimed_date)        AS first_claim,
    MAX(claimed_date)        AS last_claim,
    COUNT(DISTINCT status)   AS status_variety,
    'BONUS_MULTI_CLAIM' AS fraud_rule
FROM bonuses
GROUP BY user_id, promo_code
HAVING COUNT(bonus_id) > 1
ORDER BY total_claims DESC;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R09 — Bonus Farming (claiming across many promo codes)
-- ════════════════════════════════════════════════════════════════════════
SELECT
    user_id,
    COUNT(DISTINCT promo_code)  AS unique_promos_claimed,
    COUNT(bonus_id)             AS total_bonus_claims,
    SUM(bonus_amount)           AS total_bonus_amount,
    SUM(CASE WHEN status = 'approved' THEN bonus_amount ELSE 0 END) AS approved_bonus,
    'BONUS_FARMING' AS fraud_rule
FROM bonuses
GROUP BY user_id
HAVING COUNT(DISTINCT promo_code) > 2
    OR COUNT(bonus_id) > 3
ORDER BY total_bonus_claims DESC;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R10 — Self-Referral Fraud (same device or IP)
-- ════════════════════════════════════════════════════════════════════════
SELECT
    r.referral_id,
    r.referrer_id,
    r.referee_id,
    r.referral_date,
    r.bonus_amount,
    r.bonus_status,
    r.same_device,
    r.same_ip,
    u1.email AS referrer_email,
    u2.email AS referee_email,
    'SELF_REFERRAL' AS fraud_rule
FROM referrals r
JOIN users u1 ON r.referrer_id = u1.user_id
JOIN users u2 ON r.referee_id  = u2.user_id
WHERE r.same_device = 1
   OR r.same_ip     = 1;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R11 — Referral Ring Detection (one user referring many)
-- ════════════════════════════════════════════════════════════════════════
WITH referral_counts AS (
    SELECT
        referrer_id,
        COUNT(referee_id)                        AS total_referrals,
        SUM(bonus_amount)                        AS total_bonus_earned,
        SUM(CASE WHEN bonus_status = 'paid' THEN 1 ELSE 0 END) AS paid_referrals,
        SUM(CASE WHEN same_device = 1 OR same_ip = 1 THEN 1 ELSE 0 END) AS suspicious_referrals
    FROM referrals
    GROUP BY referrer_id
)
SELECT
    rc.*,
    u.email,
    u.registration_date,
    u.kyc_status,
    u.country,
    ROUND(CAST(suspicious_referrals AS FLOAT) / total_referrals * 100, 1) AS suspicious_pct,
    'REFERRAL_RING' AS fraud_rule
FROM referral_counts rc
JOIN users u ON rc.referrer_id = u.user_id
WHERE total_referrals > 5
   OR suspicious_referrals > 0
ORDER BY suspicious_pct DESC, total_referrals DESC;


-- ════════════════════════════════════════════════════════════════════════
-- RULE R12 — Impossible Travel / IP Velocity
-- Flags users who have transactions from two different IP addresses within 2 hours
-- ════════════════════════════════════════════════════════════════════════
SELECT
    t1.user_id,
    t1.transaction_id AS txn_1,
    t2.transaction_id AS txn_2,
    t1.transaction_date AS time_1,
    t2.transaction_date AS time_2,
    t1.ip_address AS ip_1,
    t2.ip_address AS ip_2,
    'IMPOSSIBLE_TRAVEL' AS fraud_rule
FROM transactions t1
JOIN transactions t2
    ON  t1.user_id = t2.user_id
    AND t1.transaction_id != t2.transaction_id
    AND t2.transaction_date BETWEEN t1.transaction_date AND DATETIME(t1.transaction_date, '+2 hours')
    AND t1.ip_address != t2.ip_address
ORDER BY t1.transaction_date DESC;


-- ════════════════════════════════════════════════════════════════════════
-- COMPOSITE RISK SCORE QUERY
-- Combines all rule signals into a single per-user risk score
-- ════════════════════════════════════════════════════════════════════════
WITH
dup_signals AS (
    SELECT user_id,
           MAX(CASE WHEN dup_type = 'email'  THEN 15 ELSE 0 END) +
           MAX(CASE WHEN dup_type = 'phone'  THEN 15 ELSE 0 END) +
           MAX(CASE WHEN dup_type = 'device' THEN 10 ELSE 0 END) AS dup_score
    FROM (
        SELECT u1.user_id, 'email'  AS dup_type FROM users u1 JOIN users u2 ON u1.email=u2.email     AND u1.user_id<u2.user_id
        UNION ALL
        SELECT u1.user_id, 'phone'  AS dup_type FROM users u1 JOIN users u2 ON u1.phone=u2.phone     AND u1.user_id<u2.user_id
        UNION ALL
        SELECT u1.user_id, 'device' AS dup_type FROM users u1 JOIN users u2 ON u1.device_id=u2.device_id AND u1.user_id<u2.user_id
    ) dups
    GROUP BY user_id
),
tx_signals AS (
    SELECT
        user_id,
        ROUND(AVG(CASE WHEN transaction_hour BETWEEN 0 AND 5 THEN 8.0 ELSE 0 END), 2) AS odd_hour_score,
        ROUND(AVG(CASE WHEN amount > 4000 THEN 8.0 ELSE 0 END), 2)                    AS high_amount_score,
        ROUND(AVG(CASE WHEN amount % 1000 = 0 AND amount >= 1000 THEN 4.0 ELSE 0 END), 2) AS round_amount_score,
        COUNT(*) AS tx_count
    FROM transactions
    GROUP BY user_id
),
bonus_signals AS (
    SELECT
        user_id,
        CASE WHEN COUNT(bonus_id) > 2 OR COUNT(DISTINCT promo_code) > 1 THEN 8 ELSE 0 END AS bonus_score
    FROM bonuses
    GROUP BY user_id
),
ref_signals AS (
    SELECT
        referrer_id AS user_id,
        CASE WHEN SUM(CASE WHEN same_device=1 OR same_ip=1 THEN 1 ELSE 0 END) > 0 THEN 10 ELSE 0 END AS ref_score
    FROM referrals
    GROUP BY referrer_id
),
kyc_signals AS (
    SELECT
        user_id,
        CASE WHEN kyc_status = 'rejected' THEN 12
             WHEN kyc_status = 'pending'  THEN 5
             WHEN is_verified = 0         THEN 8
             ELSE 0 END AS kyc_score
    FROM users
)
SELECT
    u.user_id,
    u.email,
    u.country,
    u.kyc_status,
    u.registration_date,
    COALESCE(ds.dup_score,   0) AS dup_score,
    COALESCE(ks.kyc_score,   0) AS kyc_score,
    COALESCE(ts.odd_hour_score + ts.high_amount_score + ts.round_amount_score, 0) AS tx_score,
    COALESCE(bs.bonus_score, 0) AS bonus_score,
    COALESCE(rs.ref_score,   0) AS ref_score,
    MIN(100,
        COALESCE(ds.dup_score,   0) +
        COALESCE(ks.kyc_score,   0) +
        COALESCE(ts.odd_hour_score + ts.high_amount_score + ts.round_amount_score, 0) +
        COALESCE(bs.bonus_score, 0) +
        COALESCE(rs.ref_score,   0)
    ) AS total_risk_score,
    CASE
        WHEN MIN(100, COALESCE(ds.dup_score,0)+COALESCE(ks.kyc_score,0)+
             COALESCE(ts.odd_hour_score+ts.high_amount_score+ts.round_amount_score,0)+
             COALESCE(bs.bonus_score,0)+COALESCE(rs.ref_score,0)) >= 65 THEN 'CRITICAL'
        WHEN MIN(100, COALESCE(ds.dup_score,0)+COALESCE(ks.kyc_score,0)+
             COALESCE(ts.odd_hour_score+ts.high_amount_score+ts.round_amount_score,0)+
             COALESCE(bs.bonus_score,0)+COALESCE(rs.ref_score,0)) >= 40 THEN 'HIGH'
        WHEN MIN(100, COALESCE(ds.dup_score,0)+COALESCE(ks.kyc_score,0)+
             COALESCE(ts.odd_hour_score+ts.high_amount_score+ts.round_amount_score,0)+
             COALESCE(bs.bonus_score,0)+COALESCE(rs.ref_score,0)) >= 20 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_tier
FROM users u
LEFT JOIN dup_signals   ds ON u.user_id = ds.user_id
LEFT JOIN tx_signals    ts ON u.user_id = ts.user_id
LEFT JOIN bonus_signals bs ON u.user_id = bs.user_id
LEFT JOIN ref_signals   rs ON u.user_id = rs.user_id
LEFT JOIN kyc_signals   ks ON u.user_id = ks.user_id
ORDER BY total_risk_score DESC;


-- ════════════════════════════════════════════════════════════════════════
-- MONITORING VIEW — Daily Fraud KPI Summary
-- Schedule this to run nightly for ops dashboard
-- ════════════════════════════════════════════════════════════════════════
SELECT
    DATE(transaction_date)                              AS txn_date,
    COUNT(*)                                            AS total_transactions,
    SUM(amount)                                         AS total_volume,
    SUM(CASE WHEN transaction_hour BETWEEN 0 AND 5 THEN 1 ELSE 0 END)  AS odd_hour_count,
    SUM(CASE WHEN amount > 4000 THEN 1 ELSE 0 END)                     AS high_value_count,
    SUM(CASE WHEN amount % 1000 = 0 AND amount >= 1000 THEN 1 ELSE 0 END) AS round_amount_count,
    SUM(CASE WHEN status = 'reversed' THEN 1 ELSE 0 END)               AS reversed_count,
    SUM(CASE WHEN is_international = 1 THEN 1 ELSE 0 END)              AS international_count,
    ROUND(
        100.0 * SUM(CASE WHEN transaction_hour BETWEEN 0 AND 5 THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                   AS odd_hour_rate_pct
FROM transactions
GROUP BY DATE(transaction_date)
ORDER BY txn_date DESC;
