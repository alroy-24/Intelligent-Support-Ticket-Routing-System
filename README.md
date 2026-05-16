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

This project is the routing layer that makes that possible.

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

> _To be filled in once models are trained. Will include:_
> - Macro-F1 per category, baseline vs. transformer
> - Ordinal urgency MAE and confusion matrix
> - Business-weighted cost reduction vs. random routing
> - P50 / P95 end-to-end latency
> - Per-1000-tickets API cost

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
- [ ] Repo scaffold + dependencies
- [ ] Data ingestion for Bitext + Twitter
- [ ] LLM labeling pipeline + validation sample
- [ ] Baseline TF-IDF + LogReg category classifier
- [ ] DistilBERT category classifier
- [ ] Ordinal urgency model
- [ ] LLM summarizer + entity extractor
- [ ] FastAPI orchestration service
- [ ] Confusion-cost matrix & business-weighted eval
- [ ] Drift monitoring job
- [ ] Active learning loop
- [ ] Streamlit demo on HF Spaces
