import unittest
from unittest.mock import patch

from nki_guidance import sdk_context


class NkiGuidanceTests(unittest.TestCase):
    def test_missing_sdk_is_explicit_not_guessed(self):
        with patch("nki_guidance.importlib.import_module", side_effect=ImportError("no nki")):
            text, metadata = sdk_context()
        self.assertFalse(metadata["installed_sdk_verified"])
        self.assertIn("unavailable locally", text)
        self.assertIn("stationary.T @ moving", text)
        self.assertIn("no profiler evidence", text)
        self.assertEqual(metadata["signatures"], {})

    def test_installed_signatures_recorded_without_loading_candidate(self):
        class FakeIsa:
            @staticmethod
            def tensor_scalar(dst, data, op0, operand0):
                pass
            dma_copy = nc_matmul = tensor_tensor = tensor_copy = tensor_scalar
        class FakeNki:
            __version__ = "test-sdk"
        with patch("nki_guidance.importlib.import_module", side_effect=[FakeNki, FakeIsa]):
            text, metadata = sdk_context()
        self.assertTrue(metadata["installed_sdk_verified"])
        self.assertEqual(metadata["nki_version"], "test-sdk")
        self.assertIn("nisa.tensor_scalar(dst, data, op0, operand0)", text)


if __name__ == "__main__":
    unittest.main()
