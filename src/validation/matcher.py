import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class Matcher:

    def __init__(

        self,

        max_distance=10.0

    ):

        self.max_distance = max_distance

    def match(

        self,

        predictions,

        targets

    ):

        if len(predictions) == 0:

            return [], [], list(range(len(targets)))

        if len(targets) == 0:

            return [], list(range(len(predictions))), []

        pred_xyz = np.array(

            [

                [

                    p.z,

                    p.y,

                    p.x

                ]

                for p in predictions

            ],

            dtype=np.float32

        )

        gt_xyz = np.array(

            [

                [

                    g.z,

                    g.y,

                    g.x

                ]

                for g in targets

            ],

            dtype=np.float32

        )

        distance_matrix = cdist(

            pred_xyz,

            gt_xyz

        )

        rows, cols = linear_sum_assignment(

            distance_matrix

        )

        matches = []

        matched_predictions = set()

        matched_targets = set()

        for r, c in zip(

            rows,

            cols

        ):

            if distance_matrix[r, c] > self.max_distance:

                continue

            matches.append(

                (

                    r,

                    c,

                    float(

                        distance_matrix[r, c]

                    )

                )

            )

            matched_predictions.add(r)

            matched_targets.add(c)

        unmatched_predictions = [

            i

            for i in range(

                len(predictions)

            )

            if i not in matched_predictions

        ]

        unmatched_targets = [

            i

            for i in range(

                len(targets)

            )

            if i not in matched_targets

        ]

        return (

            matches,

            unmatched_predictions,

            unmatched_targets

        )