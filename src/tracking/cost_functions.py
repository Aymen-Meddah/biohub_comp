import numpy as np


class CostFunctions:

    def __init__(

        self,

        distance_weight=1.0,

        embedding_weight=1.0,

        division_weight=1.0

    ):

        self.distance_weight = distance_weight

        self.embedding_weight = embedding_weight

        self.division_weight = division_weight

    def euclidean_distance(

        self,

        cell1,

        cell2

    ):

        p1 = np.array([

            cell1["z"],

            cell1["y"],

            cell1["x"]

        ], dtype=np.float32)

        p2 = np.array([

            cell2["z"],

            cell2["y"],

            cell2["x"]

        ], dtype=np.float32)

        return np.linalg.norm(p1 - p2)

    def embedding_distance(

        self,

        emb1,

        emb2

    ):

        emb1 = np.asarray(emb1)

        emb2 = np.asarray(emb2)

        return np.linalg.norm(

            emb1 - emb2

        )

    def division_cost(

        self,

        cell1,

        cell2

    ):

        if cell1.get("division", False):

            return 0.0

        return 1.0

    def total_cost(

        self,

        cell1,

        cell2

    ):

        cost = 0.0

        cost += (

            self.distance_weight *

            self.euclidean_distance(

                cell1,

                cell2

            )

        )

        if (

            "embedding" in cell1

            and

            "embedding" in cell2

        ):

            cost += (

                self.embedding_weight *

                self.embedding_distance(

                    cell1["embedding"],

                    cell2["embedding"]

                )

            )

        cost += (

            self.division_weight *

            self.division_cost(

                cell1,

                cell2

            )

        )

        return cost