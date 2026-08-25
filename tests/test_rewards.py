import math
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e2egmner.rl import GMNERReward, RewardBreakdown, compute_reward  # noqa: E402


def _response(think: str, answer: str) -> str:
    return f"<think>{think}</think><answer>{answer}</answer>"


def test_perfect_reward_follows_equation_32():
    answer = "Alice|PER|[0,0,10,10]; Bob|ORG|[20,20,30,30]"
    prediction = _response("<obj>Alice|[5,5]</obj> <obj>Bob|[25,25]</obj>", answer)
    result = compute_reward(prediction, answer)

    assert isinstance(result, RewardBreakdown)
    assert result.valid
    assert result.task_reward == 1.0
    assert result.fine_reward == 1.0
    assert result.consistency_reward == 1.0
    assert result.format_reward == 1.0
    assert result.parse_penalty == 0.0
    assert result.total == pytest.approx(1.55)
    assert result.total == pytest.approx(
        result.task_reward
        + result.fine_contribution
        + result.consistency_contribution
        + result.format_contribution
        + result.parse_penalty_contribution
    )


def test_trace_consistency_is_entity_bound_not_any_box():
    answer = "Alice|PER|[0,0,10,10]; Bob|PER|[20,20,30,30]"
    swapped = _response("<obj>Alice|[25,25]</obj> <obj>Bob|[5,5]</obj>", answer)
    result = GMNERReward()(swapped, answer)

    # Answer metrics remain perfect; both anchors belong to the wrong identity.
    assert result.task_reward == 1.0
    assert result.fine_reward == 1.0
    assert result.consistency_reward == 0.0
    assert result.total == pytest.approx(1.30)


@pytest.mark.parametrize(
    ("trace", "answer"),
    [
        ("<ungroundable>Alice</ungroundable>", "Alice|PER|[0,0,10,10]"),
        ("<obj>Bob|[5,5]</obj>", "Bob|ORG|None"),
    ],
)
def test_missed_and_forced_grounding_break_trace_answer_consistency(trace, answer):
    result = GMNERReward()(_response(trace, answer), answer)
    assert result.valid
    assert result.task_reward == 1.0
    assert result.consistency_reward == 0.0


def test_missing_and_duplicate_trace_records_use_max_inventory_denominator():
    answer = "Alice|PER|[0,0,10,10]; Bob|ORG|None"

    missing = GMNERReward()(_response("<obj>Alice|[5,5]</obj>", answer), answer)
    assert missing.trace_count == 1
    assert missing.consistency_reward == 0.5

    duplicate = GMNERReward()(
        _response(
            "<obj>Alice|[5,5]</obj> "
            "<ungroundable>Bob</ungroundable> "
            "<ungroundable>Bob</ungroundable>",
            answer,
        ),
        answer,
    )
    assert duplicate.trace_count == 3
    assert duplicate.consistency_reward == pytest.approx(2.0 / 3.0)

    conflict = GMNERReward()(
        _response(
            "<obj>Alice|[5,5]</obj> <ungroundable>Alice</ungroundable>; "
            "<ungroundable>Bob</ungroundable>",
            answer,
        ),
        answer,
    )
    assert conflict.trace_count == 3
    assert conflict.consistency_reward == pytest.approx(2.0 / 3.0)


def test_multi_box_obj_anchors_are_one_entity_and_all_must_be_inside():
    answer = "Alice|PER|[[0,0,10,10],[20,20,30,30]]"
    valid = GMNERReward()(
        _response("<obj>Alice|[5,5]</obj> <obj>Alice|[25,25]</obj>", answer),
        answer,
    )
    assert valid.trace_count == 1
    assert valid.consistency_reward == 1.0

    one_outside = GMNERReward()(
        _response("<obj>Alice|[5,5]</obj> <obj>Alice|[50,50]</obj>", answer),
        answer,
    )
    assert one_outside.trace_count == 1
    assert one_outside.consistency_reward == 0.0


def test_repeated_mentions_separated_by_reasoning_remain_distinct_entities():
    answer = "Ravens|ORG|[0,0,10,10]; Ravens|ORG|[20,20,30,30]"
    result = GMNERReward()(
        _response(
            "<obj>Ravens|[5,5]</obj> first occurrence. "
            "<obj>Ravens|[25,25]</obj> second occurrence.",
            answer,
        ),
        answer,
    )
    assert result.trace_count == 2
    assert result.consistency_reward == 1.0


@pytest.mark.parametrize(
    "malformed",
    [
        "<think><obj>Alice|[5]</obj></think><answer>Alice|PER|[0,0,10,10]</answer>",
        "<think>x</think><answer>Alice|PER|[0,0,10,10]",
        "<think>x</think><answer>Alice|PER|[10,0,0,10]</answer>",
        "<think>x</think><answer></answer>",
        "<think></think><answer>None</answer>",
        "prefix <think>x</think><answer>None</answer>",
    ],
)
def test_malformed_response_gates_every_other_component(malformed):
    result = GMNERReward()(malformed, "Alice|PER|[0,0,10,10]")
    assert not result.valid
    assert result.total == -0.20
    assert result.parse_penalty == 1.0
    assert result.format_reward == 0.0
    assert result.task_reward == 0.0
    assert result.fine_reward == 0.0
    assert result.consistency_reward == 0.0
    assert result.gmner_f1 == result.mner_f1 == result.eeg_f1 == 0.0


def test_explicit_empty_answer_uses_formula_empty_set_boundaries():
    result = GMNERReward()(_response("No licensed entity.", "None"), "<answer>None</answer>")

    assert result.valid
    assert result.pred_count == result.gold_count == result.trace_count == 0
    assert result.gmner_f1 == result.mner_f1 == result.eeg_f1 == 1.0
    assert result.count_reward == 1.0
    assert result.span_reward == 0.0
    assert result.type_reward == 0.0
    assert result.support_reward == 0.0
    assert result.localization_reward == 1.0
    assert result.fine_reward == pytest.approx(0.4)
    assert result.consistency_reward == 1.0
    assert result.total == pytest.approx(1.43)


def test_count_penalties_include_zero_count_cases_from_equation_42():
    under = GMNERReward()(_response("No entity.", "None"), "Alice|PER|[0,0,10,10]")
    over = GMNERReward()(
        _response("<obj>Alice|[5,5]</obj>", "Alice|PER|[0,0,10,10]"),
        "None",
    )
    assert under.count_reward == 0.5
    assert over.count_reward == pytest.approx(0.6)
    assert math.isfinite(under.total) and math.isfinite(over.total)
