"""Streamlit demo: paste a ticket, see the routing + urgency + summary.

Architecture: this app calls RoutingService directly (in-process), the same
orchestrator the FastAPI service uses. That keeps the demo self-contained —
one Hugging Face Space, no separate API to deploy. If you'd rather have the
demo hit a remote /route URL, swap the call site for `requests.post(...)`.

Run locally:
  streamlit run src/ticketrouting/demo/app.py

Deploy to Hugging Face Spaces:
  1. New Space → Streamlit → public.
  2. Push this repo.
  3. Add a Secret `GROQ_API_KEY` (or `ANTHROPIC_API_KEY`) in the Space settings.
  4. Make sure artifacts/ is committed (joblib models) — they're small.
  5. Add `app_file: src/ticketrouting/demo/app.py` to the Space's README.
"""
from __future__ import annotations

import asyncio

import streamlit as st
from dotenv import load_dotenv

from ticketrouting.api.service import RoutingService
from ticketrouting.schemas import RoutingDecision, TicketIn, Urgency

load_dotenv()

URGENCY_COLORS: dict[str, str] = {
    Urgency.LOW.value: "#22c55e",       # green
    Urgency.MEDIUM.value: "#eab308",    # yellow
    Urgency.HIGH.value: "#f97316",      # orange
    Urgency.CRITICAL.value: "#ef4444",  # red
}

EXAMPLE_TICKETS = [
    "My credit card was charged twice this month, please refund",
    "I can't log into my account, password reset isn't working",
    "How do I update my shipping address?",
    "The app crashes when I try to open the dashboard",
    "Could you add dark mode to the settings page?",
    "Our whole team is locked out — the SSO login is failing for everyone since this morning",
]


@st.cache_resource
def get_service() -> RoutingService:
    """Load the orchestrator once per Streamlit process (not per re-run)."""
    return RoutingService.from_env()


def _route_sync(svc: RoutingService, ticket: TicketIn) -> RoutingDecision:
    """Streamlit is sync; bridge to the async orchestrator."""
    return asyncio.run(svc.route(ticket))


def main() -> None:
    st.set_page_config(
        page_title="Ticket Routing Demo",
        page_icon="📨",
        layout="wide",
    )

    st.title("Intelligent Support Ticket Routing")
    st.caption(
        "Three models in parallel: TF-IDF category classifier, ordinal urgency "
        "scorer, and an LLM summariser + entity extractor. Paste a ticket below."
    )

    svc = get_service()

    with st.sidebar:
        st.subheader("Service health")
        for name, loaded in svc.loaded.items():
            icon = "✅" if loaded else "⚠️"
            st.write(f"{icon} **{name}** — {'loaded' if loaded else 'not loaded (fallback)'}")
        if not all(svc.loaded.values()):
            st.info(
                "Missing components return typed fallbacks rather than 500s. "
                "Train the missing model (or set an LLM API key) to enable.",
                icon="ℹ️",
            )
        st.divider()
        st.caption("See the [README](README.md) for what's built vs planned.")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader("Ticket")
        example = st.selectbox(
            "Try an example", ["(write your own)", *EXAMPLE_TICKETS], index=0
        )
        default_text = "" if example == "(write your own)" else example
        text = st.text_area(
            "Ticket text",
            value=default_text,
            height=200,
            placeholder="Paste the customer's message here...",
        )
        submit = st.button("Route this ticket", type="primary", disabled=not text.strip())

    with col_out:
        st.subheader("Routing decision")
        if submit and text.strip():
            with st.spinner("Running models…"):
                decision = _route_sync(svc, TicketIn(text=text))
            _render_decision(decision)
        else:
            st.caption("Submit a ticket to see the model output.")


def _render_decision(decision: RoutingDecision) -> None:
    cat = decision.route_to[0].value if decision.route_to else "—"
    urg = decision.urgency.value
    urg_color = URGENCY_COLORS.get(urg, "#888")

    # Headline metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Route to", cat.upper())
    m2.markdown(
        f"<div style='font-size:0.85rem;color:gray'>Urgency</div>"
        f"<div style='display:inline-block;padding:4px 12px;border-radius:6px;"
        f"background:{urg_color};color:white;font-weight:600;font-size:1.4rem;"
        f"margin-top:2px;'>{urg.upper()}</div>",
        unsafe_allow_html=True,
    )
    m3.metric("Latency", f"{decision.latency_ms} ms" if decision.latency_ms is not None else "—")

    st.divider()

    # Confidences
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Route confidence**")
        st.progress(min(max(decision.route_confidence, 0.0), 1.0))
        st.caption(f"{decision.route_confidence:.2%}")
    with c2:
        st.write("**Urgency confidence**")
        st.progress(min(max(decision.urgency_score, 0.0), 1.0))
        st.caption(f"{decision.urgency_score:.2%}")

    if decision.route_confidence < 0.6:
        st.warning(
            "Route confidence is low — in production this would escalate to "
            "human triage and feed into the active-learning loop.",
            icon="🤔",
        )

    # Summary + entities
    st.subheader("Summary")
    if decision.summary:
        st.write(decision.summary)
    else:
        st.caption(
            "No summary returned — the summariser is unavailable, rate-limited, "
            "or exceeded the per-request deadline."
        )

    if decision.entities is not None:
        st.subheader("Entities")
        e1, e2 = st.columns(2)
        with e1:
            st.write(f"**Product:** {decision.entities.product or '—'}")
            st.write(
                f"**Sentiment:** "
                f"{decision.entities.sentiment.value if decision.entities.sentiment else '—'}"
            )
        with e2:
            codes = ", ".join(decision.entities.error_codes) or "—"
            st.write(f"**Error codes:** {codes}")
            st.write(f"**Deadline mentioned:** {decision.entities.deadline_mentioned or '—'}")

    with st.expander("Raw JSON response"):
        st.json(decision.model_dump(mode="json"))


if __name__ == "__main__":
    main()
