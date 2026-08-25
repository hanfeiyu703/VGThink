from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_vgthink_data import convert_file, convert_record, main  # noqa: E402


def _record(answer: str, *, response: str | None = None, image: str = "sample.png") -> dict:
    return {
        "id": 7,
        "messages": [
            {"role": "user", "content": "Text: Alice met ACME and the Twins."},
            {
                "role": "assistant",
                "content": response if response is not None else answer,
            },
        ],
        "images": [image],
    }


def test_record_uses_the_ordered_official_answer_for_both_stages() -> None:
    answer = "Alice|PER|[10,20,30,40]; ACME|ORG|None; Twins|PER|[[0,0,20,20],[100,50,200,100]]"
    semantic_cot = "Semantic line one.\nSemantic line two."
    source = f"<think>{semantic_cot}</think><answer>{answer}</answer>"

    converted = convert_record(_record(answer, response=source), (200, 100))
    response = converted["messages"][-1]["content"]

    assert response.startswith(f"<think>{semantic_cot}\n\nEntity-centric Grounded Reasoning:")
    assert response.endswith(f"</think><answer>{answer}</answer>")
    assert (
        response.index("<obj>Alice|[20,30]</obj>")
        < response.index("<ungroundable>ACME</ungroundable>")
        < response.index("<obj>Twins|[10,10]</obj>")
    )
    assert "<obj>Twins|[10,10]</obj> <obj>Twins|[150,75]</obj>" in response

    assert converted["grounded_entity_count"] == 2
    assert [target["text"] for target in converted["region_targets"]] == [
        "Alice",
        "ACME",
        "Twins",
    ]
    assert converted["region_targets"] == [
        {
            "text": "Alice",
            "etype": "PER",
            "visible": True,
            "loss_enabled": True,
            "source": "official_answer",
            "boxes": [[0.05, 0.2, 0.15, 0.4]],
            "points": [[0.1, 0.30000000000000004]],
        },
        {
            "text": "ACME",
            "etype": "ORG",
            "visible": False,
            "loss_enabled": True,
            "source": "official_answer",
            "boxes": [],
            "points": [],
        },
        {
            "text": "Twins",
            "etype": "PER",
            "visible": True,
            "loss_enabled": True,
            "source": "official_answer",
            "boxes": [[0.0, 0.0, 0.1, 0.2], [0.5, 0.5, 1.0, 1.0]],
            "points": [[0.05, 0.1], [0.75, 0.75]],
        },
    ]


def test_none_answer_gets_an_explicit_empty_trace() -> None:
    converted = convert_record(_record("None"), (8, 6))

    assert converted["messages"][-1]["content"].endswith(
        "Trace-answer consistency: the empty inventory agrees with the final answer."
        "</think><answer>None</answer>"
    )
    assert converted["region_targets"] == []
    assert converted["grounded_entity_count"] == 0


def test_cli_reads_e2e_names_and_writes_output_atomically(tmp_path: Path) -> None:
    input_dir = tmp_path / "sft"
    output_dir = tmp_path / "prepared"
    image_root = tmp_path / "data"
    input_dir.mkdir()
    image_root.mkdir()
    Image.new("RGB", (40, 20)).save(image_root / "sample.png")
    record = _record("Alice|PER|[0,0,20,10]")
    (input_dir / "train_sft.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert (
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--image-root",
                str(image_root),
                "--splits",
                "train",
            ]
        )
        == 0
    )

    prepared = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8"))
    assert prepared["region_targets"][0]["boxes"] == [[0.0, 0.0, 0.5, 0.5]]
    assert prepared["region_targets"][0]["points"] == [[0.25, 0.25]]
    assert list(output_dir.glob("*.tmp")) == []


def test_failed_conversion_does_not_replace_an_existing_output(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (10, 10)).save(image_root / "sample.png")
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text(json.dumps(_record("not an official answer")) + "\n", encoding="utf-8")
    output_path = tmp_path / "output.jsonl"
    output_path.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot parse segment"):
        convert_file(input_path, output_path, image_root)

    assert output_path.read_text(encoding="utf-8") == "keep me\n"
    assert list(tmp_path.glob(".output.jsonl.*.tmp")) == []
