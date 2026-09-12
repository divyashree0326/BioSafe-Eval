import pandas as pd

reviewed = pd.read_csv("results_v2_reviewed.csv")
judged = pd.read_csv("results_v2_final.csv")

# Bring the L2 judge labels into the manually reviewed dataset
df = reviewed.merge(
    judged[["model", "topic", "level", "judge_label"]],
    on=["model", "topic", "level"],
    how="left"
)

# Resolve the final label:
# 1. Manual review takes priority
# 2. L2 LLM judge takes priority over the old rule-based label
# 3. Otherwise retain the rule-based label
def resolve_label(row):

    if pd.notna(row["review_note"]) and row["review_note"].strip() != "":
        # The manually reviewed L4 case was NOT actually a refusal.
        return "answered_reframed"

    if pd.notna(row["judge_label"]):
        return row["judge_label"]

    return row["label"]


df["final_label"] = df.apply(resolve_label, axis=1)

df.to_csv("results_v2_complete.csv", index=False)

print("Complete dataset saved.")
print(f"Total rows: {len(df)}")
print("\nFinal label counts:")
print(df["final_label"].value_counts().to_string())
