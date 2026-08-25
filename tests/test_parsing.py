from __future__ import annotations

import unittest

from e2egmner.evaluation.parsing import (
    extract_answer_text,
    parse_triples,
    replace_answer_text,
    split_entity_segments,
    triples_to_canon_text,
)


class StructuredAnswerParsingTest(unittest.TestCase):
    def test_html_semicolon_is_not_an_entity_separator(self) -> None:
        answer = "Fox &amp ; Fin|ORG|[1,2,3,4]; Paris|LOC|None"
        self.assertEqual(
            split_entity_segments(answer),
            ["Fox &amp ; Fin|ORG|[1,2,3,4]", "Paris|LOC|None"],
        )
        triples = parse_triples(answer, strict=True, where="test")
        self.assertEqual([triple.text for triple in triples], ["Fox &amp ; Fin", "Paris"])

    def test_multi_box_and_pipe_in_mention_round_trip(self) -> None:
        answer = "A | B|OTHER|[[1,2,3,4],[5,6,7,8]]"
        triples = parse_triples(answer, strict=True, where="test")
        self.assertEqual(triples[0].text, "A | B")
        self.assertEqual(triples_to_canon_text(triples), answer)

    def test_invalid_box_fails_strict_and_is_marked_invalid_when_lenient(self) -> None:
        answer = "Alice|PER|[10,0,0,10]"
        with self.assertRaisesRegex(ValueError, "unordered or empty"):
            parse_triples(answer, strict=True, where="test")
        parsed = parse_triples(answer, strict=False, where="test")
        self.assertEqual(len(parsed), 1)
        self.assertFalse(parsed[0].region_valid)

    def test_answer_extraction_and_replacement_preserve_reasoning(self) -> None:
        response = "<think>reason</think><answer>Alice|PER|None</answer>"
        self.assertEqual(extract_answer_text(response), "Alice|PER|None")
        self.assertEqual(
            replace_answer_text(response, "Paris|LOC|None"),
            "<think>reason</think><answer>Paris|LOC|None</answer>",
        )


if __name__ == "__main__":
    unittest.main()
