"""Map Bitext intents to our internal Category taxonomy.

The Bitext customer-support dataset uses ~27 fine-grained intents (e.g. `get_refund`,
`change_shipping_address`, `delete_account`). We collapse these into the broader team-level
categories the routing system actually decides between.

Intents that don't map cleanly fall through to `Category.OTHER` — better to be honest than
to force a label.
"""
from __future__ import annotations

from ticketrouting.schemas import Category

# Source: Bitext-customer-support-llm-chatbot-training-dataset intent column.
# Coverage is intentionally conservative: unknown intents -> OTHER.
INTENT_TO_CATEGORY: dict[str, Category] = {
    # Billing — payments, refunds, invoices, subscriptions
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
    # Account — profile, login, registration, deletion
    "create_account": Category.ACCOUNT,
    "delete_account": Category.ACCOUNT,
    "edit_account": Category.ACCOUNT,
    "switch_account": Category.ACCOUNT,
    "recover_password": Category.ACCOUNT,
    "registration_problems": Category.ACCOUNT,
    "newsletter_subscription": Category.ACCOUNT,
    # Technical — shipping/delivery logistics, addresses
    "change_shipping_address": Category.TECHNICAL,
    "set_up_shipping_address": Category.TECHNICAL,
    "delivery_options": Category.TECHNICAL,
    "delivery_period": Category.TECHNICAL,
    # Bug / complaint
    "complaint": Category.BUG,
    "review": Category.BUG,
    # Feature request / general
    "contact_customer_service": Category.OTHER,
    "contact_human_agent": Category.OTHER,
}


def map_intent(intent: str) -> Category:
    """Return the Category for a Bitext intent, defaulting to OTHER for unknowns."""
    return INTENT_TO_CATEGORY.get(intent, Category.OTHER)
