import numpy as np

from scipy.optimize import linear_sum_assignment

from src.tracking.cost_functions import CostFunctions


class HungarianMatcher:

    def __init__(self):

        self.cost_function = CostFunctions()

    def cost_matrix(

        self,

        cells_t,

        cells_t1

    ):

        matrix = np.zeros(

            (

                len(cells_t),

                len(cells_t1)

            ),

            dtype=np.float32

        )

        for i, cell1 in enumerate(cells_t):

            for j, cell2 in enumerate(cells_t1):

                matrix[i, j] = self.cost_function.total_cost(

                    cell1,

                    cell2

                )

        return matrix

    def match(

        self,

        cells_t,

        cells_t1

    ):

        if len(cells_t) == 0:

            return []

        if len(cells_t1) == 0:

            return []

        cost = self.cost_matrix(

            cells_t,

            cells_t1

        )

        rows, cols = linear_sum_assignment(

            cost

        )

        matches = []

        for r, c in zip(rows, cols):

            matches.append(

                {

                    "source": cells_t[r],

                    "target": cells_t1[c],

                    "cost": float(cost[r, c])

                }

            )

        return matches