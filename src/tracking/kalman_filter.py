import numpy as np

try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None


class CellKalmanFilter:

    def __init__(self):

        if KalmanFilter is None:
            self.filter = None
            self._position = np.zeros(3, dtype=float)
            self._velocity = np.zeros(3, dtype=float)
            return

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

        values = np.array(

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

        if self.filter is None:
            self._velocity = values[:3] - self._position
            self._position = values[:3]
            return

        self.filter.x = values

    def predict(self):

        if self.filter is None:
            return self._position.copy()

        self.filter.predict()

        return self.filter.x[:3].copy()

    def update(

        self,

        z,

        y,

        x

    ):

        if self.filter is None:
            new_position = np.array([z, y, x], dtype=float)
            self._velocity = new_position - self._position
            self._position = new_position
            return self._position.copy()

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

        if self.filter is None:
            return self._position.copy()

        return self.filter.x[:3].copy()

    @property
    def velocity(self):

        if self.filter is None:
            return self._velocity.copy()

        return self.filter.x[3:].copy()
