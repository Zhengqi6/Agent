import tempfile
import unittest
from pathlib import Path

from datasets.text_chunker import TextChunk, load_text_chunks


class TextChunkerTests(unittest.TestCase):
    def test_generates_overlapping_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = Path(tmpdir) / "sample.txt"
            sample_path.write_text("abcdefg", encoding="utf-8")
            chunks = list(load_text_chunks(str(sample_path), chunk_size=4, overlap=1))
            self.assertTrue(any(isinstance(chunk, TextChunk) for chunk in chunks))
            self.assertGreaterEqual(len(chunks), 2)
            self.assertEqual(chunks[0].text, "abcd")
            self.assertEqual(chunks[0].document_id, "sample")


if __name__ == "__main__":
    unittest.main()
