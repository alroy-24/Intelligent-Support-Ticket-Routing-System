from ticketrouting.schemas import Category, RoutingDecision, TicketIn, Urgency


def test_ticket_in_minimal():
    t = TicketIn(text="My card was charged twice")
    assert t.text.startswith("My card")
    assert t.customer_tier is None


def test_routing_decision_roundtrip():
    d = RoutingDecision(
        route_to=[Category.BILLING],
        route_confidence=0.91,
        urgency=Urgency.HIGH,
        urgency_score=0.78,
    )
    payload = d.model_dump()
    assert payload["route_to"] == ["billing"]
    assert payload["urgency"] == "high"
