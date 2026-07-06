import numpy as np


class CostMatrix:

    def __init__(

        self,

        distance_weight=1.0,

        embedding_weight=0.5,

        division_weight=0.25,

        max_distance=30.0

    ):

        self.distance_weight = distance_weight
        self.embedding_weight = embedding_weight
        self.division_weight = division_weight
        self.max_distance = max_distance

    def _distance_cost(

        self,

        node1,

        node2

    ):

        d = np.linalg.norm(

            np.array(node1.position) -

            np.array(node2.position)

        )

        if d > self.max_distance:

            return np.inf

        return d

    def _embedding_cost(

        self,

        node1,

        node2

    ):

        if node1.embedding is None:

            return 0.0

        if node2.embedding is None:

            return 0.0

        e1 = np.asarray(

            node1.embedding,

            dtype=np.float32

        )

        e2 = np.asarray(

            node2.embedding,

            dtype=np.float32

        )

        cosine = np.dot(

            e1,

            e2

        ) / (

            np.linalg.norm(e1)

            *

            np.linalg.norm(e2)

            +

            1e-8

        )

        return 1.0 - cosine

    def _division_cost(

        self,

        node1,

        node2

    ):

        return abs(

            node1.division_probability -

            node2.division_probability

        )

    def compute(

        self,

        previous_nodes,

        current_nodes

    ):

        matrix = np.zeros(

            (

                len(previous_nodes),

                len(current_nodes)

            ),

            dtype=np.float32

        )

        for i, n1 in enumerate(previous_nodes):

            for j, n2 in enumerate(current_nodes):

                distance = self._distance_cost(

                    n1,

                    n2

                )

                if np.isinf(distance):

                    matrix[i, j] = np.inf
                    continue

                embedding = self._embedding_cost(

                    n1,

                    n2

                )

                division = self._division_cost(

                    n1,

                    n2

                )

                matrix[i, j] = (

                    self.distance_weight * distance +

                    self.embedding_weight * embedding +

                    self.division_weight * division

                )

        return matrix