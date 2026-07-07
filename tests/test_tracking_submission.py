import tempfile
import unittest
from pathlib import Path

from src.inference.submission import SUBMISSION_COLUMNS, SubmissionGenerator
from src.tracking.node import Node
from src.tracking.track_manager import TrackManager


class TrackingSubmissionTests(unittest.TestCase):

    def test_track_manager_links_node_history(self):
        manager = TrackManager(max_missing=2)
        first = Node(
            node_id=1,
            dataset="sample",
            t=0,
            z=1.0,
            y=2.0,
            x=3.0,
        )
        second = Node(
            node_id=2,
            dataset="sample",
            t=1,
            z=2.0,
            y=3.0,
            x=4.0,
        )

        track = manager.create_track(first)
        manager.update_track(track.id, second)

        self.assertEqual(len(track.nodes), 2)
        self.assertEqual(first.track_id, track.id)
        self.assertEqual(second.track_id, track.id)

    def test_submission_generator_writes_expected_columns(self):
        manager = TrackManager()
        manager.create_track(
            Node(
                node_id=1,
                dataset="sample",
                t=0,
                z=1.0,
                y=2.0,
                x=3.0,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submission.csv"
            generator = SubmissionGenerator(path)
            dataframe = generator.from_tracks(manager.tracks)

            self.assertEqual(list(dataframe.columns), SUBMISSION_COLUMNS)
            self.assertTrue(generator.validate(dataframe))
            self.assertEqual(dataframe.iloc[0]["row_type"], "node")
            self.assertEqual(generator.save(manager.tracks), path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
