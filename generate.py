import random
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

def generate(rows=10_000, seed=42, locale="ar_SA"):
    random.seed(seed)
    Faker.seed(seed)
    fake = Faker(locale)

    activity_catalog = build_activity_catalog()

    # Unique numeric IDs
    verification_codes = random.sample(range(10**9, 10**10), rows)  # 10-digit
    id_numbers        = random.sample(range(10**6, 10**7), rows)    # 7-digit
    phone_numbers     = random.sample(range(10**8, 10**9), rows)    # 9-digit

    activity_durations = [
        "ساعة", "ساعتان", "3 ساعات", "4 ساعات",
        "يوم", "يومان", "3 أيام", "4 أيام", "5 أيام", "أسبوع",
        "أسبوعان", "3 أسابيع", "نصف شهر", "شهر", "شهران", "3 أشهر"
    ]

    rng = np.random.default_rng(seed)
    records = []
    for i in range(rows):
        # Base per-beneficiary fields
        verification_code = str(verification_codes[i])
        id_number = str(id_numbers[i])
        phone_number = str(phone_numbers[i])
        name = arabic_full_name(fake)
        activity_description = random_activity_description(activity_catalog)
        duration = random.choice(activity_durations)
        base_amount = random.randint(2_000, 2_000_000)

        # Multi-cycle expansion
        n = rng.integers(min_cycles, max_cycles + 1)
        start = int(rng.integers(1, 95))
        cycles = list(range(start, min(start + n, 101)))

        for c in cycles:
            noise = 1 + rng.uniform(-base_amount_noise, base_amount_noise)
            pay = max(2000, min(2_000_000, int(base_amount * noise)))
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


import argparse, sys

def main():
    p = argparse.ArgumentParser(description="Generate synthetic Arabic cash-transfer csv data.")
    p.add_argument("--rows", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", type=str, default="synthetic_cash_transfer_data.csv")
    p.add_argument("--locale", type=str, default="ar_SA")
    
    # key line in notebooks:
    args, _ = p.parse_known_args()

    df = generate(rows=args.rows, seed=args.seed, locale=args.locale)
    df.to_csv(args.csv, index=False, encoding="utf-8-sig")
    print(f"Wrote CSV: {args.csv} (rows={len(df)})")
    
    bz_df = pd.read_csv("synthetic_cash_transfer_data.csv", encoding="utf-8-sig")
    return bz_df

if __name__ == "__main__":
    main()