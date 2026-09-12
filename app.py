import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="BioSafe-Eval",
    page_icon="🧬",
    layout="wide"
)

st.title("BioSafe-Eval")

st.caption(
    "Small-scale evaluation of LLM safety consistency in biological contexts"
)

# Load data
df = pd.read_csv("results_v2_complete.csv")


# --------------------------------------------------
# Overview
# --------------------------------------------------

st.header("Evaluation Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total responses", len(df))

with col2:
    st.metric("Models evaluated", df["model"].nunique())

with col3:
    st.metric("Topics evaluated", df["topic"].nunique())

st.divider()


# --------------------------------------------------
# Main finding: L2
# --------------------------------------------------

st.header("Key Finding: L2 Boundary Handling")

st.write(
    "L2 represents abstract procedural requests. "
    "The evaluation asks whether models refuse outright or "
    "engage conceptually while explicitly withholding operational detail."
)

l2 = df[df["level"] == "L2_procedural_abstract"].copy()

l2_counts = (
    l2.groupby(["model", "final_label"])
    .size()
    .unstack(fill_value=0)
)


# Only the labels that actually occur at L2
for label in ["boundary_aware", "refused"]:
    if label not in l2_counts.columns:
        l2_counts[label] = 0

l2_counts = l2_counts[
    ["boundary_aware", "refused"]
]


# Short model names for display
display_names = {
    "openai/gpt-oss-120b": "GPT-OSS-120B",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
    "openai/gpt-oss-safeguard-20b": "GPT-OSS-Safeguard-20B"
}

l2_counts.index = [
    display_names.get(model, model)
    for model in l2_counts.index
]


# Human-readable legend labels
l2_counts = l2_counts.rename(
    columns={
        "boundary_aware": "Boundary-aware",
        "refused": "Refused"
    }
)


# --------------------------------------------------
# L2 Chart
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

l2_counts.plot(
    kind="bar",
    stacked=True,
    ax=ax
)

ax.set_title("L2 Response Behavior (n=5 prompts per model)")
ax.set_xlabel("")
ax.set_ylabel("Number of responses")
ax.set_ylim(0, 5)

ax.legend(
    title="Response behavior",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.xticks(rotation=0)
plt.tight_layout()

st.pyplot(fig)

st.caption(
    "Each model was evaluated on 5 L2 prompts. "
    "No actionable responses were observed."
)


# --------------------------------------------------
# L2 Results
# --------------------------------------------------

st.subheader("L2 Results")

l2_display = l2_counts.copy()

l2_display["Total"] = l2_display.sum(axis=1)

l2_display["Boundary-aware %"] = (
    l2_display["Boundary-aware"] /
    l2_display["Total"] * 100
).round(1)

l2_display["Refused %"] = (
    l2_display["Refused"] /
    l2_display["Total"] * 100
).round(1)

l2_display = l2_display[
    [
        "Boundary-aware",
        "Refused",
        "Total",
        "Boundary-aware %",
        "Refused %"
    ]
]

st.dataframe(
    l2_display,
    use_container_width=True
)

st.info(
    "In this sample, GPT-OSS-Safeguard-20B produced "
    "boundary-aware responses on all 5 L2 prompts, while "
    "GPT-OSS-120B and GPT-OSS-20B each refused 4 of 5."
)


# --------------------------------------------------
# Full experiment
# --------------------------------------------------

st.divider()

st.header("Behavior Across Risk Levels")

summary = (
    df.groupby(["model", "level", "final_label"])
    .size()
    .reset_index(name="count")
)

summary["model"] = summary["model"].map(
    lambda x: display_names.get(x, x)
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Response Explorer
# --------------------------------------------------

st.divider()

st.header("Response Explorer")

selected_model = st.selectbox(
    "Model",
    sorted(df["model"].unique()),
    format_func=lambda x: display_names.get(x, x)
)

selected_level = st.selectbox(
    "Risk level",
    [
        "L0_benign",
        "L1_sensitive_conceptual",
        "L2_procedural_abstract",
        "L3_explicit_boundary",
        "L4_boundary_pressure"
    ]
)

filtered = df[
    (df["model"] == selected_model) &
    (df["level"] == selected_level)
]

for _, row in filtered.iterrows():

    with st.expander(
        f"{row['topic']} — {row['final_label']}"
    ):
        st.write("Prompt:")
        st.code(row["prompt"])

        st.write("Model response:")
        st.write(row["response"])

        if (
            pd.notna(row.get("review_note"))
            and str(row["review_note"]).strip()
        ):
            st.write("Manual review note:")
            st.info(row["review_note"])