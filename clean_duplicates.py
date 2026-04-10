import pandas as pd

df = pd.read_csv("financial_documents_corrected.csv")

before = len(df)

# Keep only first occurrence
df = df.drop_duplicates(subset=["document_id"], keep="first")

after = len(df)

df.to_csv("financial_documents_corrected.csv", index=False, encoding="utf-8-sig")

print(f"Removed duplicates: {before - after}")
print(f"Remaining records: {after}")