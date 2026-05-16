from ticketrouting.data.intent_mapping import INTENT_TO_CATEGORY, map_intent
from ticketrouting.schemas import Category


def test_known_intents_map_to_expected_category():
    assert map_intent("get_refund") == Category.BILLING
    assert map_intent("delete_account") == Category.ACCOUNT
    assert map_intent("complaint") == Category.BUG


def test_unknown_intent_falls_through_to_other():
    assert map_intent("intent_that_does_not_exist") == Category.OTHER
    assert map_intent("") == Category.OTHER


def test_all_mappings_use_valid_categories():
    for intent, category in INTENT_TO_CATEGORY.items():
        assert isinstance(category, Category), f"{intent} maps to non-Category {category!r}"
