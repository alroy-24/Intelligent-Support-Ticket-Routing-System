from ticketrouting.data.intent_mapping import INTENT_TO_CATEGORY, map_intent
from ticketrouting.schemas import Category


def test_known_intents_map_to_expected_category():
    assert map_intent("get_refund") == Category.BILLING
    assert map_intent("delete_account") == Category.ACCOUNT
    # Honest mapping: complaint/review are feedback, not BUG.
    assert map_intent("complaint") == Category.OTHER
    assert map_intent("change_shipping_address") == Category.OTHER


def test_unknown_intent_falls_through_to_other():
    assert map_intent("intent_that_does_not_exist") == Category.OTHER
    assert map_intent("") == Category.OTHER


def test_all_mappings_use_valid_categories():
    for intent, category in INTENT_TO_CATEGORY.items():
        assert isinstance(category, Category), f"{intent} maps to non-Category {category!r}"


def test_bitext_does_not_populate_bug_technical_feature_request():
    """Bitext genuinely lacks data for these categories. The mapping must reflect that."""
    uncovered = {Category.BUG, Category.TECHNICAL, Category.FEATURE_REQUEST}
    populated = set(INTENT_TO_CATEGORY.values())
    assert uncovered.isdisjoint(populated), (
        f"Bitext should not populate {uncovered & populated} — see intent_mapping.py docstring"
    )
