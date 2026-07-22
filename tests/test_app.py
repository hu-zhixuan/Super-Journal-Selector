import unittest
from unittest.mock import MagicMock, patch
from app import parse_draft_input, JOURNAL_ALIASES


class TestAppFunctions(unittest.TestCase):

    def test_parse_draft_input_raw_text(self):
        text = "  This is my paper abstract.  "
        res = parse_draft_input(text, None)
        self.assertEqual(res, "This is my paper abstract.")

    def test_parse_draft_input_empty(self):
        res = parse_draft_input("", None)
        self.assertEqual(res, "")

    @patch("app.parse_docx")
    def test_parse_draft_input_docx(self, mock_docx):
        mock_docx.return_value = "Parsed docx text"
        mock_file = MagicMock()
        mock_file.name = "sample.docx"
        res = parse_draft_input("", mock_file)
        self.assertEqual(res, "Parsed docx text")
        mock_docx.assert_called_once_with("sample.docx")

    @patch("app.parse_pdf")
    def test_parse_draft_input_pdf(self, mock_pdf):
        mock_pdf.return_value = "Parsed pdf text"
        mock_file = MagicMock()
        mock_file.name = "paper.pdf"
        res = parse_draft_input("", mock_file)
        self.assertEqual(res, "Parsed pdf text")
        mock_pdf.assert_called_once_with("paper.pdf")

    def test_journal_aliases(self):
        self.assertIn("chb", JOURNAL_ALIASES)
        self.assertEqual(JOURNAL_ALIASES["chb"], "Computers in Human Behavior")


if __name__ == "__main__":
    unittest.main()
