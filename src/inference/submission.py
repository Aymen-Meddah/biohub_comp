from pathlib import Path

import pandas as pd


SUBMISSION_COLUMNS = [
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


class SubmissionGenerator:

    def __init__(self, output_path):
        self.output_path = Path(output_path)

    def from_tracks(self, tracks):
        rows = []
        next_node_id = 0

        for track in tracks:
            detections = getattr(track, "nodes", None) or getattr(
                track,
                "history",
                []
            )

            previous_node_id = None
            for detection in detections:
                node_id = next_node_id
                next_node_id += 1

                rows.append(
                    self._node_row(
                        node_id=node_id,
                        detection=detection,
                    )
                )

                if previous_node_id is not None:
                    rows.append(
                        self._edge_row(
                            source_id=previous_node_id,
                            target_id=node_id,
                        )
                    )

                previous_node_id = node_id

        return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)

    def save(self, tracks):
        dataframe = self.from_tracks(tracks)
        self.validate(dataframe)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(self.output_path, index=False)
        return self.output_path

    @staticmethod
    def validate(dataframe):
        missing = [
            column
            for column in SUBMISSION_COLUMNS
            if column not in dataframe.columns
        ]
        if missing:
            raise ValueError(
                f"Submission is missing required columns: {missing}"
            )

        if dataframe[SUBMISSION_COLUMNS].isnull().any().any():
            raise ValueError("Submission contains null values.")

        return True

    @staticmethod
    def _node_row(node_id, detection):
        return {
            "row_type": "node",
            "node_id": int(node_id),
            "t": int(_value(detection, "t", 0)),
            "z": float(_value(detection, "z")),
            "y": float(_value(detection, "y")),
            "x": float(_value(detection, "x")),
            "source_id": -1,
            "target_id": -1,
        }

    @staticmethod
    def _edge_row(source_id, target_id):
        return {
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": int(source_id),
            "target_id": int(target_id),
        }


def _value(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)
