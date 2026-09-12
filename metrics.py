import pandas as pd

df = pd.read_csv("results_v2_complete.csv")

# --------------------------------------------------
# 1. Counts by model and level
# --------------------------------------------------

counts = (
    df.groupby(["model", "level", "final_label"])
      .size()
      .unstack(fill_value=0)
)

print("\n=== COUNTS BY MODEL AND LEVEL ===\n")
print(counts.to_string())


# --------------------------------------------------
# 2. Percentages by model and level
# --------------------------------------------------

percentages = (
    df.groupby(["model", "level"])["final_label"]
      .value_counts(normalize=True)
      .mul(100)
      .round(1)
      .unstack(fill_value=0)
)

print("\n=== PERCENTAGES BY MODEL AND LEVEL ===\n")
print(percentages.to_string())


# --------------------------------------------------
# 3. L2 comparison
# --------------------------------------------------

l2 = df[df["level"] == "L2_procedural_abstract"]

l2_summary = (
    l2.groupby(["model", "final_label"])
      .size()
      .unstack(fill_value=0)
)

print("\n=== L2 COMPARISON ===\n")
print(l2_summary.to_string())


# --------------------------------------------------
# 4. L2 percentages
# --------------------------------------------------

l2_percentages = (
    l2.groupby("model")["final_label"]
      .value_counts(normalize=True)
      .mul(100)
      .round(1)
      .unstack(fill_value=0)
)

print("\n=== L2 PERCENTAGES ===\n")
print(l2_percentages.to_string())


# --------------------------------------------------
# 5. Save metric tables
# --------------------------------------------------

counts.to_csv("metrics_counts.csv")
percentages.to_csv("metrics_percentages.csv")
l2_summary.to_csv("metrics_l2_counts.csv")
l2_percentages.to_csv("metrics_l2_percentages.csv")

print("\nMetric files saved.")
