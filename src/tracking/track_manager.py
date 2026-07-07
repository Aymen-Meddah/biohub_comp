from src.tracking.track import Track
from src.tracking.kalman_filter import CellKalmanFilter


class TrackManager:

    def __init__(

        self,

        max_missing=5,
        max_missed=None

    ):

        if max_missed is not None:
            max_missing = max_missed

        self.max_missing = max_missing

        self.tracks = []

        self.next_track_id = 0

    def create_track(

        self,

        node

    ):

        track = Track(
            self.next_track_id,
            node
        )

        kf = CellKalmanFilter()

        z, y, x = self._coordinates(node)

        kf.initialize(
            z,
            y,
            x
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

        if isinstance(track, int):
            track = self.get_track(track)

        track.kalman.update(

            *self._coordinates(node)

        )

        track.add_node(

            node

        )

        track.missing = 0

    def predict_tracks(self):

        for track in self.tracks:

            if track.kalman is not None:
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

    def all_tracks(self):
        return list(self.tracks)

    def get_track(self, track_id):
        for track in self.tracks:
            if track.id == track_id:
                return track
        raise KeyError(f"Track not found: {track_id}")

    def mark_missed(self, track_id):
        for track in self.tracks:
            if track.id == track_id:
                track.mark_missed()
                track.missing = track.missed
                return track
        raise KeyError(f"Track not found: {track_id}")

    @staticmethod
    def _coordinates(node):
        if isinstance(node, dict):
            return node["z"], node["y"], node["x"]
        return node.z, node.y, node.x
