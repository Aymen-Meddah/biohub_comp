from src.tracking.track import Track
from src.tracking.kalman_filter import CellKalmanFilter


class TrackManager:

    def __init__(

        self,

        max_missing=5

    ):

        self.max_missing = max_missing

        self.tracks = []

        self.next_track_id = 0

    def create_track(

        self,

        node

    ):

        track = Track(

            self.next_track_id

        )

        track.add_node(

            node

        )

        kf = CellKalmanFilter()

        kf.initialize(

            node.z,

            node.y,

            node.x

        )

        track.kalman = kf

        track.missing = 0

        self.tracks.append(track)

        self.next_track_id += 1

        return track

    def update_track(

        self,

        track,

        node

    ):

        track.kalman.update(

            node.z,

            node.y,

            node.x

        )

        track.add_node(

            node

        )

        track.missing = 0

    def predict_tracks(self):

        for track in self.tracks:

            track.kalman.predict()

            track.missing += 1

    def remove_dead_tracks(self):

        self.tracks = [

            track

            for track in self.tracks

            if track.missing <= self.max_missing

        ]

    def active_tracks(self):

        return [

            track

            for track in self.tracks

            if track.missing == 0

        ]