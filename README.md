# BioSafe-Eval

A small-scale evaluation of LLM safety consistency across risk levels in biological contexts.

[Live dashboard](https://biosafe-eval.streamlit.app/)

---

## What this is

BioSafe-Eval is a small-scale implementation exercise using established evaluation patterns — notably [HarmBench](https://arxiv.org/abs/2402.04249) and [WMDP](https://www.wmdp.ai/) — to build, run, and analyze an LLM safety evaluation pipeline end to end. It is not a novel benchmark or jailbreak study; the goal was to understand a real evaluation methodology hands-on, not to produce a publishable safety finding.

## Methodology

- **5 topics**, spanning general immunology/virology concepts (antibody function, gain-of-function research, viral mutation, pathogen recognition, bacteria vs. viruses)
- **5 risk levels** per topic, from benign to adversarial:
  - `L0` — benign educational request
  - `L1` — sensitive conceptual discussion (dual-use/biosecurity implications, still educational)
  - `L2` — abstract procedural request, using a bracketed placeholder (`[RESTRICTED PROCEDURE]`) instead of any real technical target
  - `L3` — explicit request for a full actionable protocol (same placeholder)
  - `L4` — the L3 request wrapped in a fictional-framing / instruction-override attempt
- **3 models** tested via the Groq API: GPT-OSS-120B, GPT-OSS-20B, GPT-OSS-Safeguard-20B
- **75 total responses** (5 topics × 5 levels × 3 models)

**No real biological, chemical, or procedural hazard content was ever generated or requested.** L2–L4 prompts use a literal bracketed placeholder in place of any actual restricted content — the evaluation measures how models respond to the *framing* of a request, not to real technical specifics.

### Scoring pipeline
1. **Rule-based classifier** — keyword/length heuristics sort responses into `refused` / `answered`, with a confidence flag.
2. **LLM-as-judge** — a second model re-classifies only the ambiguous/low-confidence cases, using a 3-way rubric (`refused` / `boundary-aware` / `actionable`) that the rule-based layer structurally can't distinguish.
3. **Manual review** — one case flagged by neither layer was caught on inspection and hand-corrected (see Development History).

## Key finding

At the extremes (L0/L1 and L3/L4), all three models behaved almost identically — near-universal compliance on benign/conceptual prompts, near-universal refusal on explicit/adversarial ones.

The interesting divergence was at **L2** (abstract procedural requests, n=5 per model):

| Model | Boundary-aware | Refused |
|---|---:|---:|
| GPT-OSS-120B | 1/5 | 4/5 |
| GPT-OSS-20B | 1/5 | 4/5 |
| GPT-OSS-Safeguard-20B | 5/5 | 0/5 |

GPT-OSS-Safeguard-20B never refused outright at L2 — it consistently gave a structured, high-level conceptual outline while explicitly stating it was withholding operational detail. The other two models mostly refused outright instead of engaging conceptually. In this sample, the models differed in how they handled *ambiguous-risk* requests, even though their behavior was consistent at the benign and explicit/adversarial extremes.

**Sample size caveat:** these are 5-prompt-per-model results, not a validated benchmark. The pattern held consistently across all 5 topics for Safeguard-20B, which is a real signal at this scale — but "100%" here means 5/5, not confidence at scale.

## Development history

This project went through three real methodological corrections worth documenting:

1. **First prompt set was too benign.** An initial 20-prompt run (different phrasings of ordinary biology questions) produced 60/60 "answered" responses across all three models — a baseline sanity check, not a safety measurement, since nothing in that set approached any refusal-worthy boundary. The prompt set was redesigned around explicit risk levels (above) to introduce real behavioral variance.
2. **The rule-based classifier couldn't distinguish compliance from boundary-aware behavior.** Early L2 results were flagged as "answered" by keyword matching alone, which missed that several of those responses explicitly stated they were staying conceptual and withholding detail. This led to building the LLM-as-judge step with a finer-grained rubric specifically to catch this distinction.
3. **One apparent under-refusal at L4 was traced to a prompt-template flaw, not a safety failure.** The L4 template (`[RESTRICTED PROCEDURE]` + fictional framing) didn't map coherently onto one topic ("how the immune system recognizes pathogens") that has no natural "procedure" to withhold. The model responded with standard fictional-framed immunology content rather than granting an override — a case where a placeholder failed to function as intended for a mechanism-based topic, not a model safety lapse. This was caught on manual review and relabeled `refused_via_reframing`.

## Limitations

- Small sample (n=5 per model per level); not statistically powered
- Tests response to explicit bracketed placeholders, not realistic-phrasing adversarial jailbreaks — by design, to avoid generating any real sensitive content
- Single model used as LLM-judge; no cross-validation against a second judge
- Manual review covered only flagged/disagreement cases, not the full dataset
- No causal claim is made about *why* models differ (e.g., "Safeguard-20B is safer because of X training method") — only the observed behavioral difference is reported

## Repository structure
```
prompts_v2.py           # risk-level prompt generation
run_experiment.py       # API calls, raw response collection
analyze_results_v2.py   # rule-based classifier
llm_judge.py            # LLM-as-judge for ambiguous cases
merge_results.py        # combines manual review + judge output into final labels
metrics.py              # summary statistics
app.py                  # Streamlit dashboard
results_v2_complete.csv # final labeled dataset
```

## Prior work referenced

- Mazeika et al., [HarmBench](https://arxiv.org/abs/2402.04249) (2024)
- [WMDP](https://www.wmdp.ai/) — Center for AI Safety
- [JailbreakBench](https://jailbreakbench.github.io/)
- Li et al., [Multi-Turn Human Jailbreaks (MHJ)](https://arxiv.org/abs/2408.15221), Scale AI (2024)
