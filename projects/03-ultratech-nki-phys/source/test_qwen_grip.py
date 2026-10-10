import unittest

from qwen_grip import extract_source, same_program


class QwenGenerationTests(unittest.TestCase):
    def test_duplicate_detection_ignores_comments_but_not_algorithm_changes(self):
        self.assertTrue(same_program("x = 1\n", "# revised\nx=1\n"))
        self.assertFalse(same_program("x = 1\n", "x = 2\n"))
    def test_complete_code_block_can_be_preserved_without_execution(self):
        source = "def contact_batch(a_transposed, bias, initial, alpha, steps):\n    return initial\n"
        self.assertEqual(extract_source("```python\n" + source + "```"), source)

    def test_incorrect_entry_and_fragmented_answer_rejected(self):
        for content in ("", "def wrong():\n    pass", "```python\nx=1\n```\n```python\ny=2\n```"):
            with self.assertRaises((ValueError, SyntaxError)):
                extract_source(content)


if __name__ == "__main__":
    unittest.main()
