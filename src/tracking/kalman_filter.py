import numpy as np

from filterpy.kalman import KalmanFilter


class CellKalmanFilter:

    def __init__(self):

        self.filter = KalmanFilter(

            dim_x=6,

            dim_z=3

        )

        self.filter.F = np.array(

            [

                [1,0,0,1,0,0],

                [0,1,0,0,1,0],

                [0,0,1,0,0,1],

                [0,0,0,1,0,0],

                [0,0,0,0,1,0],

                [0,0,0,0,0,1]

            ],

            dtype=float

        )

        self.filter.H = np.array(

            [

                [1,0,0,0,0,0],

                [0,1,0,0,0,0],

                [0,0,1,0,0,0]

            ],

            dtype=float

        )

        self.filter.P *= 100.0

        self.filter.R *= 0.5

        self.filter.Q *= 0.01

    def initialize(

        self,

        z,

        y,

        x

    ):

        self.filter.x = np.array(

            [

                z,

                y,

                x,

                0,

                0,

                0

            ],

            dtype=float

        )

    def predict(self):

        self.filter.predict()

        return self.filter.x[:3].copy()

    def update(

        self,

        z,

        y,

        x

    ):

        self.filter.update(

            np.array(

                [

                    z,

                    y,

                    x

                ],

                dtype=float

            )

        )

        return self.filter.x[:3].copy()

    @property
    def position(self):

        return self.filter.x[:3].copy()

    @property
    def velocity(self):

        return self.filter.x[3:].copy()