import numpy as np

from scipy.optimize import linear_sum_assignment

from src.tracking.cost_matrix import CostMatrix


class HungarianTracker:

    def __init__(

        self,

        max_cost=30.0

    ):

        self.max_cost = max_cost

        self.cost_matrix = CostMatrix(

            max_distance=max_cost

        )

    def associate(

        self,

        previous_nodes,

        current_nodes

    ):

        if len(previous_nodes) == 0:

            return [], [], list(range(len(current_nodes)))

        if len(current_nodes) == 0:

            return [], list(range(len(previous_nodes))), []

        cost = self.cost_matrix.compute(

            previous_nodes,

            current_nodes

        )

        rows, cols = linear_sum_assignment(

            cost

        )

        matches = []

        matched_previous = set()

        matched_current = set()

        for r, c in zip(rows, cols):

            if np.isinf(cost[r, c]):

                continue

            if cost[r, c] > self.max_cost:

                continue

            matches.append(

                (

                    previous_nodes[r],

                    current_nodes[c],

                    float(cost[r, c])

                )

            )

            matched_previous.add(r)

            matched_current.add(c)

        unmatched_previous = [

            previous_nodes[i]

            for i in range(len(previous_nodes))

            if i not in matched_previous

        ]

        unmatched_current = [

            current_nodes[i]

            for i in range(len(current_nodes))

            if i not in matched_current

        ]

        return (

            matches,

            unmatched_previous,

            unmatched_current

        )