"""Map Bitext intents to our internal Category taxonomy.

**Honest assessment of coverage:**

Bitext is a templated e-commerce support dataset with 27 intents across ~27k examples.
It covers our production taxonomy unevenly:

  Category         | Bitext coverage | Notes
  -----------------+-----------------+----------------------------------------------------
  BILLING          | Strong          | refunds, invoices, payments, orders (~12k examples)
  ACCOUNT          | Strong          | create/delete/edit, password, registration (~7k)
  OTHER            | Strong          | feedback, contact-support, shipping (~8k)
  TECHNICAL        | None            | no engineering-issue intents in the dataset
  BUG              | None            | "complaint"/"review" are feedback, not defects
  FEATURE_REQUEST  | None            | no feature-suggestion intents in the dataset

The earlier mapping forced complaint/review → BUG and shipping → TECHNICAL to fill
categories that Bitext doesn't actually cover. That made the baseline's 99.8% F1
misleading: the model wasn't learning what "BUG" means, it was learning what Bitext's
"complaint" template looks like.

This honest mapping gives an empty BUG/TECHNICAL/FEATURE_REQUEST in training. The
baseline can't predict those classes — and that's exactly what we want it to surface,
because the Twitter dataset (planned next) is what will fill those gaps.

Unknown intents fall through to OTHER. Better to be honest than to force a label.
"""
from __future__ import annotations

from ticketrouting.schemas import Category

# Source: Bitext-customer-support-llm-chatbot-training-dataset, 27 intents total.
INTENT_TO_CATEGORY: dict[str, Category] = {
    # ── Billing: payments, refunds, invoices, orders ───────────────────────────
    "get_refund": Category.BILLING,
    "check_refund_policy": Category.BILLING,
    "track_refund": Category.BILLING,
    "check_invoice": Category.BILLING,
    "get_invoice": Category.BILLING,
    "check_payment_methods": Category.BILLING,
    "payment_issue": Category.BILLING,
    "check_cancellation_fee": Category.BILLING,
    "cancel_order": Category.BILLING,
    "change_order": Category.BILLING,
    "place_order": Category.BILLING,
    "track_order": Category.BILLING,
    # ── Account: profile, login, registration, lifecycle ───────────────────────
    "create_account": Category.ACCOUNT,
    "delete_account": Category.ACCOUNT,
    "edit_account": Category.ACCOUNT,
    "switch_account": Category.ACCOUNT,
    "recover_password": Category.ACCOUNT,
    "registration_problems": Category.ACCOUNT,
    "newsletter_subscription": Category.ACCOUNT,
    # ── Other: shipping/logistics, customer feedback, generic contact requests ─
    # NOTE: shipping intents previously mapped to TECHNICAL — wrong. Logistics
    # isn't engineering. They go to OTHER (in a real org: a fulfillment/ops queue).
    "change_shipping_address": Category.OTHER,
    "set_up_shipping_address": Category.OTHER,
    "delivery_options": Category.OTHER,
    "delivery_period": Category.OTHER,
    # NOTE: complaint/review previously mapped to BUG — wrong. These are
    # customer-feedback templates, not technical defect reports.
    "complaint": Category.OTHER,
    "review": Category.OTHER,
    # Contact-support intents — by definition the user hasn't said what they want.
    "contact_customer_service": Category.OTHER,
    "contact_human_agent": Category.OTHER,
}


def map_intent(intent: str) -> Category:
    """Return the Category for a Bitext intent, defaulting to OTHER for unknowns."""
    return INTENT_TO_CATEGORY.get(intent, Category.OTHER)
