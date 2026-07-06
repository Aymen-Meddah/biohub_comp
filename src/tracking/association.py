from src.tracking.hungarian_matcher import HungarianMatcher


class Association:

    def __init__(

        self,

        max_cost=20.0

    ):

        self.max_cost = max_cost

        self.matcher = HungarianMatcher()

    def associate(

        self,

        tracks,

        detections

    ):

        if len(tracks) == 0:

            return [], [], detections

        if len(detections) == 0:

            return [], tracks, []

        track_cells = []

        for track in tracks:

            track_cells.append(

                {

                    "id": track.id,

                    "z": track.z,

                    "y": track.y,

                    "x": track.x,

                    "embedding": track.embedding

                }

            )

        matches = self.matcher.match(

            track_cells,

            detections

        )

        assigned_tracks = set()

        assigned_detections = set()

        valid_matches = []

        for match in matches:

            if match["cost"] > self.max_cost:

                continue

            valid_matches.append(match)

            assigned_tracks.add(

                match["source"]["id"]

            )

            assigned_detections.add(

                match["target"]["id"]

            )

        unmatched_tracks = []

        for track in tracks:

            if track.id not in assigned_tracks:

                unmatched_tracks.append(track)

        unmatched_detections = []

        for detection in detections:

            if detection["id"] not in assigned_detections:

                unmatched_detections.append(

                    detection

                )

        return (

            valid_matches,

            unmatched_tracks,

            unmatched_detections

        )