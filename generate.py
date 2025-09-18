
import argparse
import random
import numpy as np
import pandas as pd
from faker import Faker


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


def random_activity_description(activity_catalog):
    category, sublist = random.choice(activity_catalog)
    sub = random.choice(sublist)
    return f"{category} - {sub}."


def arabic_full_name(fake):
    first = random.choice([fake.first_name_male, fake.first_name_female])()
    last = fake.last_name()
    return f"{first} {last}"


def generate(
    beneficiaries: int = 10_000,
    seed: int = 42,
    locale: str = "ar_SA",
    min_cycles: int = 2,
    max_cycles: int = 5,
    base_amount_noise: float = 0.20,
):
    """
    Generate a multi-cycle dataset.

    beneficiaries: number of unique beneficiaries (each expands to 2 - 5 cycles).
    """
    assert 1 <= min_cycles <= max_cycles <= 100, "min_cycles/max_cycles must be between 1 and 100"

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

    # To ensure we can always pick a valid start so that start + n - 1 <= 100
    max_start_for_min = 101 - min_cycles

    for i in range(beneficiaries):
        verification_code = str(verification_codes[i])
        id_number = str(id_numbers[i])
        phone_number = str(phone_numbers[i])

        name = arabic_full_name(fake)
        activity_description = random_activity_description(activity_catalog)
        duration = random.choice(activity_durations)

        base_amount = random.randint(2_000, 2_000_000)

        # Draw number of cycles for this beneficiary
        n_cycles = int(rng.integers(min_cycles, max_cycles + 1))
        
        # We don't want to exceed cycle 100
        max_start = 101 - n_cycles
        if max_start < 1:
            max_start = 1
        start = int(rng.integers(1, max_start + 1))
        cycles = list(range(start, start + n_cycles))  # all <= 100 by construction

        for c in cycles:
            noise = 1 + float(rng.uniform(-base_amount_noise, base_amount_noise))
            pay = int(base_amount * noise)
            pay = min(max(pay, 2_000), 2_000_000)
            records.append({
                "verification_code": verification_code,
                "beneficiary_names": name,
                "payment_amount": pay,
                "activity_desc": activity_description,
                "activty_duration": duration,
                "id_number": id_number,
                "phone_number": phone_number,
                "payment_cycle": c,
            })

    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser(description="Generate synthetic Arabic cash-transfer data (CSV only, multi-cycle).")
    
    # Keeping rows for backward compatibility
    p.add_argument("--beneficiaries", type=int, default=10_000, help="Number of unique beneficiaries (default: 10000).")
    p.add_argument("--rows", type=int, help="Alias for --beneficiaries (backward compatibility).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--csv", type=str, default="synthetic_cash_transfer_data.csv",
                   help="Output CSV file (default: synthetic_cash_transfer_data.csv).")
    p.add_argument("--locale", type=str, default="ar_SA", help="Faker locale (default: ar_SA).")
    p.add_argument("--min_cycles", type=int, default=2, help="Minimum cycles per beneficiary (default: 2).")
    p.add_argument("--max_cycles", type=int, default=5, help="Maximum cycles per beneficiary (default: 5).")
    p.add_argument("--noise", type=float, default=0.20, help="Per-cycle amount noise ±fraction (default: 0.20).")

    # parse_known_args makes this work in notebooks too
    args, _ = p.parse_known_args()

    beneficiaries = args.beneficiaries if args.rows is None else args.rows
    df = generate(
        beneficiaries=beneficiaries,
        seed=args.seed,
        locale=args.locale,
        min_cycles=args.min_cycles,
        max_cycles=args.max_cycles,
        base_amount_noise=args.noise,
    )

    # Save CSV with BOM for Arabic compatibility in Excel
    df.to_csv(args.csv, index=False, encoding="utf-8-sig")

    # Printing Plain ASCII to avoid Windows console encoding issues
    approx = int(df.shape[0])
    print(f"Wrote CSV: {args.csv}  rows={approx}  beneficiaries={beneficiaries}  "
          f"avg_cycles≈{round(approx/beneficiaries, 2)}")


if __name__ == "__main__":
    main()
