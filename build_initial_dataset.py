import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

COMPANIES = [
    "ABC Traders", "XYZ Company", "Sun Holdings", "Metro Lanka", "Prime Solutions",
    "සහන් ට්‍රේඩර්ස්", "ග්‍රීන් මාර්ට්", "නව වෙළඳසැල", "සිංහ වෙළඳ සමාගම", "ලංකා සපයන්නෝ"
]

SUPPLIERS = [
    "Lanka Foods", "Metro Suppliers", "Tech World", "City Office Supplies", "Digital Hub",
    "ලංකා වෙළඳ සමාගම", "නව ආහාර සපයන්නෝ", "ඩිජිටල් උපකරණ", "කාර්යාල උපකරණ", "සීග්‍ර සැපයුම්"
]

DOCUMENT_TYPES = ["invoice", "receipt", "purchase_order", "delivery_note"]

ENGLISH_ITEMS = [
    "Rice Bags", "Printer Paper", "USB Devices", "Pens", "Keyboards",
    "Monitors", "Chairs", "Tables", "Files", "Mouse Devices"
]

SINHALA_ITEMS = [
    "මුද්‍රණ කඩදාසි", "සීනි පැකට්", "යතුරුපුවරු", "මේස", "පුටු",
    "ලිපිගොනු", "පරිගණක මූසික", "කාර්යාල පොත්", "පෑන්", "ආහාර පැකට්"
]

STATUSES = ["paid", "unpaid", "pending", "delivered"]


def random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")


def build_dataset(num_rows=3000):
    rows = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 3, 1)

    for i in range(1, num_rows + 1):
        language = random.choice(["english", "sinhala"])
        document_type = random.choice(DOCUMENT_TYPES)
        company_name = random.choice(COMPANIES)
        supplier_name = random.choice(SUPPLIERS)
        item_description = random.choice(ENGLISH_ITEMS if language == "english" else SINHALA_ITEMS)

        quantity = random.randint(1, 50)
        unit_price = random.choice([50, 100, 120, 150, 220, 500, 850, 1000, 1200, 2500, 5000, 8500])
        final_total_amount = quantity * unit_price
        payable_amount = 0 if document_type == "receipt" and random.choice([True, False]) else final_total_amount

                rows.append({
            "document_id": f"DOC{i:05d}",
            "document_type": document_type,
            "company_name": company_name,
            "supplier_name": supplier_name,
            "date": random_date(start_date, end_date),
            "item_description": item_description,
            "quantity": quantity,
            "unit_price": unit_price,
            "raw_total_amount": final_total_amount,
            "final_total_amount": final_total_amount,
            "total_status": "valid",
            "payable_amount": payable_amount,
            "currency": "LKR",
            "status": random.choice(STATUSES),
            "language": language,
            "raw_text": "",
            "corrected_text": "",
            "source_json": "",
            "correction_log": "",
            "correction_confidence": 1.0,
            "items_json": "",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset(num_rows=3000)
    df.to_csv("financial_documents_corrected.csv", index=False, encoding="utf-8-sig")
    print("Created financial_documents_corrected.csv with", len(df), "rows")