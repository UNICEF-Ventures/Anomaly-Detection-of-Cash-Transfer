#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Arabic cash-transfer dataset generator (CSV only, multi-cycle, with anomalies).

Schema per row:
- verification_code   (beneficiary ID, 10-digit string, reused across cycles)
- beneficiary_names   (Arabic first + last name)
- payment_amount      (2,000 to 2,000,000)
- activity_desc       (Arabic "<category> - <sub>.")
- activty_duration    (short Arabic time phrase)
- id_number           (7-digit string, per beneficiary)
- phone_number        (9-digit string, per beneficiary)
- payment_cycle       (integer 1..100)

Anomalies (not labeled in output; only simulated):
- amount_spike
- frequency_surge
- inconsistent_description
- amount_service_mismatch
"""

import argparse
import random
import numpy as np
import pandas as pd
from faker import Faker


# ---------------------------
# Activity catalog & helpers
# ---------------------------

def build_activity_catalog():
    return [
        ("الخدمة الطبية", [
            "فحص طبي عام", "تطعيم الأطفال", "استشارة طبية", "صرف الأدوية",
            "متابعة الحوامل", "خدمات الإسعاف الأولية", "تحاليل مخبرية أساسية"
        ]),
        ("الخدمة التعليمية", [
            "حصة محو الأمية", "دعم دراسي للتلاميذ", "دورة مهارات رقمية",
            "توعية مدرسية بالصحة", "تدريب للمعلمين", "تقوية في الرياضيات والعلوم"
        ]),
        ("المساعدة الصحية", [
            "جلسة توعية صحية", "متابعة التغذية للأطفال", "دعم نفسي اجتماعي",
            "توزيع حزم النظافة", "رعاية منزلية للحالات المزمنة", "إحالة إلى مركز صحي"
        ]),
        ("صيانة المرافق", [
            "إصلاح شبكة المياه", "صيانة دورات المياه", "ترميم عيادة",
            "استبدال مضخة", "إصلاح مولد كهربائي", "صيانة خزان المياه"
        ]),
    ]


def random_activity_desc(activity_catalog):
    category, sublist = random.choice(activity_catalog)
    sub = random.choice(sublist)
    return f"{category} - {sub}."


def arabic_full_name(fake):
    first = random.choice([fake.first_name_male, fake.first_name_female])()
    last = fake.last_name()
    return f"{first} {last}"


# ---------------------------
# Anomaly injection
# ---------------------------

def inject_anomalies(df, min_rate=0.05, max_rate=0.10, seed=123):
    """
    Inject four anomaly types per payment cycle:

    1. amount_spike:
       payment_amount is strongly above/below the beneficiary's historical mean.

    2. frequency_surge:
       >3 rows for the same (verification_code, payment_cycle) by adding extra rows.

    3. inconsistent_description:
       activity_desc replaced with mismatching free-text phrases.

    4. amount_service_mismatch:
       payment_amount far from the typical amount for that service (activity_desc).

    Internally uses a helper column '_anomaly_tag' but drops it before returning.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Helper tag column (not returned)
    df["_anomaly_tag"] = "normal"

    # Per-beneficiary mean for amount_spike
    mean_per_ben = df.groupby("verification_code")["payment_amount"].mean().to_dict()
    # Per-service mean for amount_service_mismatch
    mean_per_service = df.groupby("activity_desc")["payment_amount"].mean().to_dict()

    # Pool of inconsistent/mismatching Arabic descriptions
    inconsistent_desc_pool = [
        "دفع غير مطابق للوصف.",
        "خدمة غير معروفة في هذا البرنامج.",
        "وصف غير متسق مع المبلغ.",
        "معاملة بدون وصف واضح.",
        "دفع استثنائي خارج النمط المعتاد.",
        "وصف لا يتطابق مع سجل المستفيد.",
    ]

    extra_records = []

    for cycle in sorted(df["payment_cycle"].unique()):
        idx_cycle = df.index[df["payment_cycle"] == cycle].tolist()
        n_c = len(idx_cycle)
        if n_c == 0:
            continue

        # Choose rate between min_rate and max_rate for this cycle
        rate_c = float(rng.uniform(min_rate, max_rate))
        target_total = int(round(rate_c * n_c))

        # Ensure at least 4 anomalies if we have enough rows (so we can represent 4 types)
        if n_c >= 4:
            target_total = max(4, target_total)
        else:
            target_total = max(1, target_total)

        # Don’t try to modify more rows than we have
        target_total = min(target_total, n_c)

        if target_total <= 0:
            continue

        # Roughly equal anomalies across 4 types in this cycle
        base_per_type = max(1, target_total // 4)
        counts = {
            "amount_spike": base_per_type,
            "inconsistent_description": base_per_type,
            "amount_service_mismatch": base_per_type,
            "frequency_surge": base_per_type,
        }
        # Adjust total if rounding left some slack
        assigned = 4 * base_per_type
        if assigned > target_total:
            # Reduce one from frequency_surge if we overshot
            diff = assigned - target_total
            counts["frequency_surge"] = max(1, counts["frequency_surge"] - diff)
        elif assigned < target_total:
            # Add extra anomalies to frequency_surge if we undershot
            counts["frequency_surge"] += (target_total - assigned)

        # We only *modify* a subset of rows; frequency_surge will also add new rows
        normal_idxs = idx_cycle.copy()

        # ---------- 1. Amount Spike ----------
        k_spike = min(counts["amount_spike"], len(normal_idxs))
        if k_spike > 0:
            spike_idx = rng.choice(normal_idxs, size=k_spike, replace=False)
            for idx in spike_idx:
                vc = df.at[idx, "verification_code"]
                base = mean_per_ben.get(vc, df.at[idx, "payment_amount"])
                if pd.isna(base) or base <= 0:
                    base = max(df.at[idx, "payment_amount"], 2000)

                # stronger spikes to make them very visible
                if rng.random() < 0.5:
                    new_amt = base * float(rng.uniform(3.5, 5.0))  # high spike
                else:
                    new_amt = base * float(rng.uniform(0.05, 0.25))  # low spike

                new_amt = int(np.clip(new_amt, 2000, 2_000_000))
                df.at[idx, "payment_amount"] = new_amt
                df.at[idx, "_anomaly_tag"] = "amount_spike"
            normal_idxs = [i for i in normal_idxs if i not in spike_idx]

        # ---------- 2. Inconsistent Description ----------
        k_desc = min(counts["inconsistent_description"], len(normal_idxs))
        if k_desc > 0:
            desc_idx = rng.choice(normal_idxs, size=k_desc, replace=False)
            for idx in desc_idx:
                df.at[idx, "activity_desc"] = rng.choice(inconsistent_desc_pool)
                df.at[idx, "_anomaly_tag"] = "inconsistent_description"
            normal_idxs = [i for i in normal_idxs if i not in desc_idx]

        # ---------- 3. Amount–Service Mismatch ----------
        k_mismatch = min(counts["amount_service_mismatch"], len(normal_idxs))
        if k_mismatch > 0:
            mismatch_idx = rng.choice(normal_idxs, size=k_mismatch, replace=False)
            for idx in mismatch_idx:
                desc = df.at[idx, "activity_desc"]
                svc_mean = mean_per_service.get(desc, df.at[idx, "payment_amount"])
                if pd.isna(svc_mean) or svc_mean <= 0:
                    svc_mean = max(df.at[idx, "payment_amount"], 2000)

                # Make amount clearly off relative to typical service mean
                if rng.random() < 0.5:
                    new_amt = svc_mean * float(rng.uniform(3.0, 5.0))   # much higher than normal for this service
                else:
                    new_amt = svc_mean * float(rng.uniform(0.05, 0.3))  # much lower than normal for this service

                new_amt = int(np.clip(new_amt, 2000, 2_000_000))
                df.at[idx, "payment_amount"] = new_amt
                df.at[idx, "_anomaly_tag"] = "amount_service_mismatch"
            normal_idxs = [i for i in normal_idxs if i not in mismatch_idx]

        # ---------- 4. Frequency Surge ----------
        # We try to create some frequency_surge anomalies by adding extra rows
        k_freq = counts["frequency_surge"]
        if k_freq > 0 and len(normal_idxs) > 0:
            cycle_df = df.loc[idx_cycle]
            eligible_codes = (
                cycle_df[cycle_df["_anomaly_tag"] == "normal"]["verification_code"].unique()
            )
            if len(eligible_codes) > 0:
                chosen_codes = rng.choice(
                    eligible_codes,
                    size=min(k_freq, len(eligible_codes)),
                    replace=False,
                )
                for vc in chosen_codes:
                    base_rows = df[
                        (df["verification_code"] == vc) &
                        (df["payment_cycle"] == cycle)
                    ]
                    if base_rows.empty:
                        continue
                    base_row = base_rows.iloc[0].to_dict()
                    base_idx = base_rows.index[0]

                    # Mark the original row too (group-level anomaly)
                    if df.at[base_idx, "_anomaly_tag"] == "normal":
                        df.at[base_idx, "_anomaly_tag"] = "frequency_surge"

                    # Add 3 extra similar rows for this beneficiary/cycle
                    for _ in range(3):
                        new_row = base_row.copy()
                        amt = new_row["payment_amount"]
                        if pd.isna(amt) or amt <= 0:
                            amt = mean_per_ben.get(vc, 20_000)
                        new_amt = int(np.clip(
                            amt * float(rng.uniform(0.8, 1.2)),
                            2000,
                            2_000_000
                        ))
                        new_row["payment_amount"] = new_amt
                        new_row["_anomaly_tag"] = "frequency_surge"
                        extra_records.append(new_row)

    if extra_records:
        df = pd.concat([df, pd.DataFrame(extra_records)], ignore_index=True)

    # Drop helper tag before returning – no labels in final data
    df.drop(columns=["_anomaly_tag"], inplace=True, errors="ignore")
    return df


# ---------------------------
# Main generator
# ---------------------------

def generate(
    beneficiaries: int = 10_000,
    seed: int = 42,
    locale: str = "ar_SA",
    min_cycles: int = 2,
    max_cycles: int = 5,
    base_amount_noise: float = 0.20,
    min_anom_rate: float = 0.05,
    max_anom_rate: float = 0.10,
):
    """
    Generate a multi-cycle dataset with injected anomalies.

    beneficiaries: number of unique beneficiaries (each expands to 2–5 cycles).
    min_anom_rate, max_anom_rate: per-cycle anomaly rate range (e.g. 0.05–0.10).
    """
    assert 1 <= min_cycles <= max_cycles <= 100, \
        "min_cycles/max_cycles must be between 1 and 100"

    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)
    fake = Faker(locale)

    activity_catalog = build_activity_catalog()

    # Unique identifiers per beneficiary
    verification_codes = random.sample(range(10**9, 10**10), beneficiaries)  # 10-digit
    id_numbers        = random.sample(range(10**6, 10**7), beneficiaries)    # 7-digit
    phone_numbers     = random.sample(range(10**8, 10**9), beneficiaries)    # 9-digit

    activity_durations = [
        "ساعة", "ساعتان", "3 ساعات", "4 ساعات",
        "يوم", "يومان", "3 أيام", "4 أيام", "5 أيام", "أسبوع",
        "أسبوعان", "3 أسابيع", "نصف شهر", "شهر", "شهران", "3 أشهر"
    ]

    rng = np.random.default_rng(seed)
    records = []

    for i in range(beneficiaries):
        verification_code = str(verification_codes[i])
        id_number = str(id_numbers[i])
        phone_number = str(phone_numbers[i])

        name = arabic_full_name(fake)
        activity_desc = random_activity_desc(activity_catalog)
        duration = random.choice(activity_durations)

        base_amount = random.randint(2_000, 2_000_000)

        # Draw number of cycles for this beneficiary
        n_cycles = int(rng.integers(min_cycles, max_cycles + 1))

        # Ensure we don't exceed cycle 100
        max_start = 101 - n_cycles
        if max_start < 1:
            max_start = 1
        start = int(rng.integers(1, max_start + 1))
        cycles = list(range(start, start + n_cycles))

        for c in cycles:
            noise = 1 + float(rng.uniform(-base_amount_noise, base_amount_noise))
            pay = int(base_amount * noise)
            pay = min(max(pay, 2_000), 2_000_000)
            records.append({
                "verification_code": verification_code,
                "beneficiary_names": name,
                "payment_amount": pay,
                "activity_desc": activity_desc,
                "activty_duration": duration,
                "id_number": id_number,
                "phone_number": phone_number,
                "payment_cycle": c,
            })

    df = pd.DataFrame(records)
    df = inject_anomalies(
        df,
        min_rate=min_anom_rate,
        max_rate=max_anom_rate,
        seed=seed + 1,
    )
    return df


# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate synthetic Arabic cash-transfer data (CSV, multi-cycle, with anomalies)."
    )

    # Backward-compatible: --rows as alias for --beneficiaries
    p.add_argument("--beneficiaries", type=int, default=10_000,
                   help="Number of unique beneficiaries (default: 10000).")
    p.add_argument("--rows", type=int,
                   help="Alias for --beneficiaries (backward compatibility).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    p.add_argument("--csv", type=str, default="synthetic_cash_transfer_data.csv",
                   help="Output CSV file (default: synthetic_cash_transfer_data.csv).")
    p.add_argument("--locale", type=str, default="ar_SA",
                   help="Faker locale (default: ar_SA).")
    p.add_argument("--min_cycles", type=int, default=2,
                   help="Minimum cycles per beneficiary (default: 2).")
    p.add_argument("--max_cycles", type=int, default=5,
                   help="Maximum cycles per beneficiary (default: 5).")
    p.add_argument("--noise", type=float, default=0.20,
                   help="Per-cycle base amount noise ±fraction (default: 0.20).")
    p.add_argument("--min_anom", type=float, default=0.05,
                   help="Minimum anomaly rate per cycle (default: 0.05).")
    p.add_argument("--max_anom", type=float, default=0.10,
                   help="Maximum anomaly rate per cycle (default: 0.10).")

    args, _ = p.parse_known_args()

    beneficiaries = args.beneficiaries if args.rows is None else args.rows

    df = generate(
        beneficiaries=beneficiaries,
        seed=args.seed,
        locale=args.locale,
        min_cycles=args.min_cycles,
        max_cycles=args.max_cycles,
        base_amount_noise=args.noise,
        min_anom_rate=args.min_anom,
        max_anom_rate=args.max_anom,
    )

    df.to_csv(args.csv, index=False, encoding="utf-8-sig")

    approx = int(df.shape[0])
    print(
        f"Wrote CSV: {args.csv}  "
        f"rows={approx}  beneficiaries={beneficiaries}  "
        f"avg_cycles≈{round(approx / beneficiaries, 2)}"
    )


if __name__ == "__main__":
    main()
