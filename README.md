# Intelligent Support Ticket Routing System

An end-to-end NLP system that auto-routes customer support tickets to the right team, scores urgency, and produces an agent-facing summary — so triage time drops from minutes to seconds.

> **Status:** in development. See [Roadmap](#roadmap) for what's built vs. planned.

---

## 1. The business problem

A mid-sized SaaS company receives **10,000+ support tickets per week**. Today:

- A human triager spends **~4 minutes per ticket** deciding who should handle it.
- Roughly **18% of tickets get misrouted**, costing another reassignment cycle.
- **Critical incidents** (outages, security, churning enterprise accounts) sit in the queue alongside low-priority feature requests.
- Agents open each ticket cold — no context, no extracted entities, no summary.

**Target outcome:** cut average time-to-first-response by 40%, reduce misrouting below 5%, and surface critical tickets within 60 seconds of arrival.

This project is the routing layer that makes that possible

---

## 2. System design

```
                    ┌─────────────────────────────────────────────┐
                    │           Raw ticket (text + metadata)       │
                    └────────────────────┬────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │         Preprocessing & embedding cache      │
                    └─────┬──────────────┬───────────────┬────────┘
                          │              │               │
                          ▼              ▼               ▼
                 ┌────────────────┐ ┌────────────┐ ┌──────────────┐
                 │  Model 1:      │ │ Model 2:   │ │ Model 3:     │
                 │  Category      │ │ Urgency    │ │ Summarizer + │
                 │  classifier    │ │ scorer     │ │ entity NER   │
                 │  (multi-label) │ │ (ordinal)  │ │ (LLM)        │
                 └───────┬────────┘ └─────┬──────┘ └──────┬───────┘
                         │                │               │
                         └────────────────┼───────────────┘
                                          ▼
                    ┌─────────────────────────────────────────────┐
                    │  Routing decision + priority + summary JSON  │
                    └─────────────────────────────────────────────┘
                                          │
                                          ▼
                              FastAPI service / Streamlit demo
```

### The three models

| Model | Task | Approach | Why |
|---|---|---|---|
| **Category** | Multi-label classification across teams (Billing, Technical, Account, Bug, Feature, etc.) | TF-IDF + LogReg baseline → fine-tuned DistilBERT | A ticket can hit multiple teams; baseline keeps us honest about whether the transformer is actually earning its latency cost. |
| **Urgency** | Ordinal classification: Low / Medium / High / Critical | Encoder + ordinal-aware loss (CORAL or weighted CE with distance penalties) | "Critical predicted as Low" is far worse than "Critical predicted as High" — plain classification ignores that asymmetry. |
| **Summary + entities** | 2-sentence summary, product names, error codes, sentiment, deadline mentions | LLM via API with JSON-mode structured output | Generative summarization is the one place an LLM clearly beats a fine-tuned small model on quality-per-engineering-hour. |

### Orchestration

A FastAPI service receives the raw ticket, runs the three models (Models 1 & 2 in parallel, Model 3 conditional on confidence), and returns:

```json
{
  "route_to": ["billing", "account"],
  "route_confidence": 0.87,
  "urgency": "high",
  "urgency_score": 0.72,
  "summary": "Enterprise customer on the Scale plan reports duplicate charges...",
  "entities": {
    "product": "Scale plan",
    "error_codes": [],
    "sentiment": "frustrated",
    "deadline_mentioned": "before end of billing cycle"
  },
  "latency_ms": 340
}
```

---

## 3. Data & labeling

**Primary dataset:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — ~3M real tweets between users and major brands. Noisy, multi-brand, realistic.

**Secondary dataset:** [Bitext customer support intents](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — cleaner labeled intents, used for sanity-checking the baseline.

### Labeling strategy

The Twitter dataset has no category or urgency labels. The pipeline:

1. **Synthetic labeling with an LLM.** Carefully-prompted Claude generates category and urgency labels for a stratified sample of ~20k tickets.
2. **Human validation.** A 500-ticket sample is hand-labeled to estimate label noise. Categories with >15% disagreement get prompt refinement and a second pass.
3. **Active learning loop.** Once the model is trained, lowest-confidence predictions are surfaced for human review and fed back into training.

**Why document this:** synthetic labeling with human validation is what real teams do. The interesting question isn't "did you label data," it's "do you know where your labels lie to you."

---

## 4. Modeling choices & tradeoffs

### Always have a baseline
TF-IDF + Logistic Regression runs in milliseconds and gives you a number to beat. If DistilBERT only beats it by 2 macro-F1 points, the latency tradeoff probably isn't worth it in production.

### Ordinal loss for urgency
Treating urgency as 4-class classification throws away the ordering. We use CORAL-style cumulative logits so the model learns that High is closer to Critical than Low is.

### LLM for summarization, not classification
LLMs are expensive and slow for high-volume classification. They shine for the open-ended summary + entity step where a fine-tuned small model would need its own labeled corpus.

### Cost control
LLM responses are cached by `sha256(ticket_text)`. Identical tickets (very common in support — same template complaints) cost nothing on the second call.

---

## 5. What makes this system production-grade

- **Confusion-cost matrix.** Errors are weighted by business impact (misrouting billing → engineering costs ~15 min; missing a critical outage costs much more). The model is tuned against this matrix, not raw accuracy.
- **Drift monitoring.** Predictions are logged with embeddings; weekly job computes category-distribution drift and embedding drift, alerts on shifts.
- **Active learning loop.** Lowest-confidence predictions surface for human review — closes the loop from production back into training data.
- **Latency & cost analysis.** P50/P95 latency and per-1000-tickets cost are tracked alongside accuracy.
- **Live demo.** Streamlit UI on Hugging Face Spaces / Modal — paste a ticket, see the routing, urgency, and summary in real time.

---

## 6. Results

### Baseline category classifier (TF-IDF + LogReg on Bitext)

| Metric | Value |
|---|---|
| Macro F1 | **0.998** |
| Weighted F1 | 0.998 |
| Training rows | 21,497 (80% of 26,872) |
| Test rows | 5,375 |
| Train time | ~5 seconds |

Per-class F1: `billing 0.998`, `account 0.998`, `other 0.999`.

### ⚠ Why this number is misleading — and why that's the most interesting result

A 99.8% macro-F1 on a 3-class problem looks like a finished system. It isn't. Two things are happening:

**1. Bitext is templated, so the classifier memorises templates, not concepts.**
Every Bitext intent uses a small set of generated phrasings ("I want help to...", "how can I..."). TF-IDF picks up on the templates trivially. The model isn't learning what "billing" *means* — it's learning what Bitext's billing template *looks like*.

**2. Bitext only honestly covers 3 of our 6 production categories.**
After auditing the intent-to-category mapping (see [intent_mapping.py](src/ticketrouting/data/intent_mapping.py)), the honest picture is:

| Category | Bitext examples | Notes |
|---|---:|---|
| BILLING | 11,927 | strong |
| ACCOUNT | 6,985 | strong |
| OTHER | 7,960 | strong (incl. shipping, feedback, contact-support) |
| TECHNICAL | 0 | dataset has no engineering-issue intents |
| BUG | 0 | "complaint"/"review" are feedback, not defects |
| FEATURE_REQUEST | 0 | dataset has no feature-suggestion intents |

The previous version of this project forced `complaint → BUG` and `change_shipping_address → TECHNICAL` to make the headline "6-class classifier" claim. That was wrong, and it inflated the per-class F1 by giving the classifier perfectly-templated phantom data. The current mapping reflects what Bitext actually contains.

**3. Out-of-distribution probe: the model collapses on real-sounding text.**

The held-out test set is in-distribution Bitext. To get a meaningful signal on whether the model *generalises*, I scored five hand-written tickets that look like real customer messages:

| Ticket | Predicted | Confidence | OK? |
|---|---|---:|---|
| "My credit card was charged twice this month" | billing | 0.49 | ✓ (low confidence on an easy billing ticket) |
| "I can't log into my account, password reset isn't working" | account | 0.99 | ✓ |
| "How do I update my shipping address?" | other | 0.99 | ✓ |
| "The app crashes when I try to open the dashboard" | account | 0.48 | ✗ — no BUG class exists in training |
| "Could you add dark mode to the settings page?" | account | 0.47 | ✗ — no FEATURE_REQUEST class exists in training |

The confidence drop from ~0.99 (in-distribution) to ~0.48 (out-of-distribution) is the model telling us it's unsure — and we should listen to it. Production routing decisions below ~0.6 confidence should escalate to human triage. This will become the threshold that drives the active-learning loop.

### Takeaway for the rest of the project

The baseline isn't "done at 99.8%." It's a working tool that:
- Sets a real floor for the transformer to beat **on Bitext** (any honest comparison must be on the same data).
- Surfaces low-confidence predictions to drive human relabelling — exactly the active-learning loop the system needs.
- Tells us, unambiguously, that **we need the Twitter dataset** to cover BUG / TECHNICAL / FEATURE_REQUEST. No amount of better modelling fixes a coverage gap in the training data.

### Twitter ingestion validates the coverage gap

The Twitter pipeline (`scripts/build_twitter_dataset.py`) downloads the Customer Support on Twitter dataset, filters to inbound customer-originated tweets, anonymises mentions/URLs, and labels categories via a Groq-served LLM. A 100-row smoke run gives:

| Category | n  | What this confirms |
|---|---:|---|
| OTHER | 66 | Twitter is noisy — lots of generic complaints and chatter. Expected. |
| BILLING | 13 | Bitext also covers this; cross-source validation. |
| BUG | 9 | **Net-new** — Bitext has zero. Real examples: "fix the accidental FaceTime call notification", "mark as read doesn't work". |
| TECHNICAL | 5 | **Net-new**. Examples: "CoD WW2 servers are a mess", "my internet has been down for an hour". |
| ACCOUNT | 4 | Cross-source validation. |
| FEATURE_REQUEST | 3 | **Net-new**. Examples: "no parental controls for explicit lyrics", "combine baggage in one booking". |

17% of the sample lands in the three categories Bitext genuinely can't reach. The full ~20k run is the next thing to kick off — the pipeline is sha256-cached so partial runs resume for free.

### Still to fill in

- DistilBERT vs. baseline on Bitext (same split)
- Both models on the Twitter dataset (real-world OOD comparison)
- Ordinal urgency MAE and confusion matrix
- Business-weighted cost reduction vs. random routing
- P50 / P95 end-to-end latency
- Per-1000-tickets API cost

---

## 7. Deployment architecture

- **API:** FastAPI + Uvicorn, containerized.
- **Models:** Category/urgency loaded in-process; LLM via API with response caching in Redis.
- **Async:** Summarization runs as a background task and is patched into the response if it finishes within budget, otherwise returned via a follow-up webhook.
- **Demo target:** Hugging Face Spaces (Streamlit) for the public demo; Modal for serverless GPU inference if needed.

---

## 8. Limitations & what's next

- **Synthetic labels carry the LLM's biases.** Validation sample estimates noise but doesn't eliminate it.
- **Twitter ≠ enterprise tickets.** Style and length differ from real B2B support. A short fine-tuning pass on a Zendesk-like corpus would help.
- **No multilingual support yet.** ~30% of real support volume is non-English; current pipeline is English-only.
- **Urgency is hard.** Even human triagers disagree ~25% of the time on High vs. Critical. The model's ceiling is bounded by label agreement.

**Next iterations:** multilingual encoder (XLM-R), few-shot examples for the LLM step keyed off retrieved similar tickets, A/B testing harness against the current routing baseline.

---

## 9. Repo layout

```
ticketrouting/
├── README.md                 # you are here
├── pyproject.toml            # dependencies & tooling
├── data/                     # raw and processed datasets (gitignored)
│   ├── raw/
│   └── processed/
├── notebooks/                # EDA, labeling experiments, error analysis
├── src/ticketrouting/
│   ├── data/                 # loaders, preprocessing, labeling
│   ├── models/
│   │   ├── category/         # multi-label classifier
│   │   ├── urgency/          # ordinal scorer
│   │   └── summary/          # LLM summarizer + entity extraction
│   ├── api/                  # FastAPI service
│   ├── eval/                 # confusion-cost matrix, drift, metrics
│   └── demo/                 # Streamlit app
├── tests/
└── scripts/                  # one-off CLI utilities (label, train, evaluate)
```

---

## 10. Roadmap

- [x] Project framing & README
- [x] Repo scaffold + dependencies
- [x] Bitext loader + honest intent→category mapping
- [x] LLM labeling pipeline (Groq / Anthropic via shared `LLMClient` protocol, sha256-cached)
- [x] Baseline TF-IDF + LogReg category classifier *(0.998 macro-F1 on Bitext — see Results for caveats)*
- [x] Twitter dataset ingestion + LLM-based category labeling *(pipeline + 100-row smoke test; ~20k full run pending)*
- [ ] Full 20k Twitter labeling run + merge with Bitext for combined training set
- [ ] DistilBERT category classifier (compare against baseline on same split)
- [x] Ordinal urgency model *(TF-IDF + Frank & Hall K-1-threshold LogReg; MAE + off-by-≥2 reporting; awaiting real urgency-labeled training set)*
- [x] LLM summarizer + entity extractor *(JSON-mode call returning summary + pydantic Entities; sha256-cached; smoke-tested on real ticket)*
- [ ] FastAPI orchestration service
- [ ] Confusion-cost matrix & business-weighted eval
- [ ] Drift monitoring job
- [ ] Active learning loop (driven by confidence threshold from baseline)
- [ ] Streamlit demo on HF Spaces
