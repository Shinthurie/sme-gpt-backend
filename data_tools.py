import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "financial_documents_corrected.csv"
CHARTS_DIR = "charts"


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"{DATASET_PATH} not found.")
    return pd.read_csv(DATASET_PATH)


def dataset_summary(df):
    return df.describe(include="all")


def column_names(df):
    return df.columns.tolist()


def detect_column_types(df):
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def extract_column_from_query(query, available_columns):
    q = query.lower()
    for col in available_columns:
        if col.lower() in q:
            return col
    return None


def extract_company_from_query(query, available_companies):
    q = query.lower()
    for company in available_companies:
        if str(company).lower() in q:
            return company
    return None


def payable_lookup(df, company_name):
    filtered = df[df["company_name"].astype(str).str.lower() == company_name.lower()]

    if filtered.empty:
        return f"No records found for company '{company_name}'."

    unpaid_docs = filtered[filtered["status"].astype(str).str.lower() == "unpaid"]
    total_payable = unpaid_docs["payable_amount"].sum()

    currency = "LKR"
    if "currency" in unpaid_docs.columns and not unpaid_docs.empty:
        currency_values = unpaid_docs["currency"].dropna().unique().tolist()
        if len(currency_values) == 1:
            currency = currency_values[0]

    return {
        "company_name": company_name,
        "all_records_found": int(len(filtered)),
        "unpaid_documents": int(len(unpaid_docs)),
        "total_payable": float(total_payable),
        "currency": currency
    }
def search_documents(df, company_name=None, document_type=None):
    filtered = df.copy()

    if company_name:
        filtered = filtered[filtered["company_name"].astype(str).str.lower() == company_name.lower()]

    if document_type:
        filtered = filtered[filtered["document_type"].astype(str).str.lower() == document_type.lower()]

    if filtered.empty:
        return "No matching documents found."

    return filtered[[
        "document_id", "document_type", "company_name", "supplier_name",
        "date", "item_description", "raw_total_amount",
        "final_total_amount", "total_status", "payable_amount", "status"
    ]].head(20)


def corrected_totals_only(df):
    filtered = df[df["total_status"].astype(str).str.lower() == "corrected"]

    if filtered.empty:
        return "No corrected total records found."

    return filtered[[
        "document_id", "company_name", "document_type",
        "quantity", "unit_price", "raw_total_amount",
        "final_total_amount", "total_status"
    ]].head(20)


def generate_basic_insights(df):
    insights = []

    if "document_type" in df.columns and not df["document_type"].dropna().empty:
        insights.append(f"Most common document type: {df['document_type'].mode()[0]}")

    if "company_name" in df.columns:
        insights.append(f"Number of unique companies: {df['company_name'].nunique()}")

    if "payable_amount" in df.columns:
        insights.append(f"Total payable amount: {df['payable_amount'].sum()}")

    if "total_status" in df.columns:
        corrected_count = (df["total_status"].astype(str).str.lower() == "corrected").sum()
        insights.append(f"Number of corrected totals: {corrected_count}")

    if "status" in df.columns and not df["status"].dropna().empty:
        insights.append(f"Most common status: {df['status'].mode()[0]}")

    return insights


def create_histogram(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    os.makedirs(CHARTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()

    filename = os.path.join(CHARTS_DIR, f"{column}_hist.png")
    plt.savefig(filename)
    plt.close()

    return filename


def create_bar_chart(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    os.makedirs(CHARTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.countplot(x=df[column])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Count of {column}")
    plt.tight_layout()

    filename = os.path.join(CHARTS_DIR, f"{column}_bar.png")
    plt.savefig(filename)
    plt.close()

    return filename