from data_tools import (
    load_dataset,
    payable_lookup,
    corrected_totals_only,
    generate_basic_insights,
    create_histogram,
    create_bar_chart
)

df = load_dataset()

print("=== PAYABLE LOOKUP ===")
print(payable_lookup(df, "ABC Traders"))

print("\n=== CORRECTED TOTALS ===")
print(corrected_totals_only(df))

print("\n=== INSIGHTS ===")
print(generate_basic_insights(df))

print("\n=== HISTOGRAM ===")
print(create_histogram(df, "final_total_amount"))

print("\n=== BAR CHART ===")
print(create_bar_chart(df, "document_type"))