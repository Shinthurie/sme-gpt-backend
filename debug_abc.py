from data_tools import load_dataset

df = load_dataset()

abc = df[df["company_name"].astype(str).str.lower() == "abc traders"]

print("Total records for ABC Traders:", len(abc))
print("\nStatuses:")
print(abc["status"].value_counts())

print("\nPayable sum of all ABC records:")
print(abc["payable_amount"].sum())

print("\nPayable sum of unpaid ABC records:")
print(abc[abc["status"].astype(str).str.lower() == "unpaid"]["payable_amount"].sum())

print("\nLast 5 ABC records:")
print(abc[["document_id", "date", "final_total_amount", "payable_amount", "status"]].tail())
