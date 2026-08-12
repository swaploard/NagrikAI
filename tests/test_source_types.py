from __future__ import annotations

from nagrik_ai.utils.source_types import (
    SOURCE_TYPE_PRIORITY,
    authority_bonus,
    authority_rank,
    classify_source_type,
)


class TestClassifySourceType:
    def test_explicit_source_type_field_takes_priority(self) -> None:
        metadata = {"title": "Some random title", "source_id": "x", "source_type": "rules"}
        assert classify_source_type(metadata) == "rules"

    def test_explicit_invalid_source_type_falls_back_to_heuristics(self) -> None:
        metadata = {"title": "FAQs_GST", "source_type": "not-a-valid-type"}
        assert classify_source_type(metadata) == "faq"

    def test_faq_by_title_prefix(self) -> None:
        assert classify_source_type({"title": "FAQs_GST_Returns"}) == "faq"
        assert classify_source_type({"title": "FAQs"}) == "faq"

    def test_manual_by_title_prefix(self) -> None:
        assert classify_source_type({"title": "Manual_Returns_Filing"}) == "user_guide"

    def test_notification_and_circular(self) -> None:
        assert classify_source_type({"title": "GST Notification 01/2024"}) == "notification"
        assert classify_source_type({"title": "GST Circular 179"}) == "circular"

    def test_rules_and_act(self) -> None:
        assert classify_source_type({"title": "GST Rules 2017"}) == "rules"
        assert classify_source_type({"title": "The Income-tax Act"}) == "act"

    def test_domain_hints(self) -> None:
        assert classify_source_type({"domain": "tutorial.gst.gov.in"}) == "user_guide"
        assert classify_source_type({"domain": "userguide.abc.com"}) == "user_guide"
        assert classify_source_type({"domain": "cbt.gst.gov.in"}) == "user_guide"
        assert classify_source_type({"domain": "help.incometax.gov.in"}) == "help"

    def test_falls_back_to_other(self) -> None:
        assert classify_source_type({"title": "Welcome", "domain": "example.com"}) == "other"
        assert classify_source_type({}) == "other"


class TestAuthorityRank:
    def test_authority_rank_order(self) -> None:
        assert authority_rank("act") < authority_rank("rules")
        assert authority_rank("rules") < authority_rank("notification")
        assert authority_rank("faq") < authority_rank("user_guide")
        assert authority_rank("help") < authority_rank("other")

    def test_unknown_type_ranks_last(self) -> None:
        assert authority_rank("unknown_type") == len(SOURCE_TYPE_PRIORITY) - 1


class TestAuthorityBonus:
    def test_default_bonus_map(self) -> None:
        assert authority_bonus("act") == authority_bonus("rules")
        assert authority_bonus("act") > authority_bonus("notification")
        assert authority_bonus("faq") > authority_bonus("other")
        assert authority_bonus("help") == 0.0

    def test_custom_bonus_map(self) -> None:
        custom = {"act": 0.2, "faq": 0.01}
        assert authority_bonus("act", custom) == 0.2
        assert authority_bonus("faq", custom) == 0.01
        assert authority_bonus("missing_type", custom) == 0.0
