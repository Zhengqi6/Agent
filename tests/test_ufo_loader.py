import tempfile
import unittest
from pathlib import Path

from datasets.ufo_loader import UfoSighting, load_ufo_dataset


class UfoLoaderTests(unittest.TestCase):
    def test_load_ufo_dataset_reads_rows(self) -> None:
        csv_content = "Date_time,city,state,country,shape,duration,comments,latitude,longitude\n"
        csv_content += (
            "1/1/2000 00:00,roswell,nm,us,disk,30 sec,"
            "Saw a bright disk overhead,33.3943,-104.5230\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "ufo_sample.csv"
            dataset_path.write_text(csv_content, encoding="utf-8")

            rows = list(load_ufo_dataset(str(dataset_path)))
            self.assertEqual(len(rows), 1)
            sighting = rows[0]
            self.assertIsInstance(sighting, UfoSighting)
            self.assertEqual(sighting.city, "roswell")
            self.assertEqual(sighting.shape, "disk")


if __name__ == "__main__":
    unittest.main()
