import numpy as np

from src.tracking.motion_model import MotionModel


class CandidateGenerator:

    def __init__(
        self,
        max_distance=25.0
    ):

        self.max_distance = max_distance

        self.motion = MotionModel()

    def distance(
        self,
        prediction,
        detection
    ):

        p = np.array(

            [

                prediction["z"],

                prediction["y"],

                prediction["x"]

            ],

            dtype=np.float32

        )

        d = np.array(

            [

                detection["z"],

                detection["y"],

                detection["x"]

            ],

            dtype=np.float32

        )

        return np.linalg.norm(
            p - d
        )

    def generate(
        self,
        tracks,
        detections
    ):

        candidates = {}

        for track in tracks:

            prediction = self.motion.predict(
                track
            )

            candidates[track.id] = []

            for detection in detections:

                dist = self.distance(

                    prediction,

                    detection

                )

                if dist <= self.max_distance:

                    candidates[
                        track.id
                    ].append(

                        {

                            "detection": detection,

                            "distance": float(dist)

                        }

                    )

        return candidates