from src.tracking.track_manager import TrackManager
from src.tracking.association import Association
from src.tracking.graph_builder import GraphBuilder


class TrackerPipeline:

    def __init__(
        self,
        max_missed=5,
        max_cost=20.0
    ):

        self.track_manager = TrackManager(
            max_missed=max_missed
        )

        self.association = Association(
            max_cost=max_cost
        )

        self.graph = GraphBuilder()

    def initialize(
        self,
        detections
    ):

        for detection in detections:

            track = self.track_manager.create_track(
                detection
            )

            self.graph.add_node(

                track.id,

                detection["t"],

                detection["z"],

                detection["y"],

                detection["x"]

            )

    def step(
        self,
        detections
    ):

        active_tracks = self.track_manager.active_tracks()

        matches, lost_tracks, new_detections = self.association.associate(

            active_tracks,

            detections

        )

        for match in matches:

            track_id = match["source"]["id"]

            detection = match["target"]

            self.track_manager.update_track(

                track_id,

                detection

            )

            self.graph.add_node(

                track_id,

                detection["t"],

                detection["z"],

                detection["y"],

                detection["x"]

            )

            self.graph.add_edge(

                track_id,

                track_id

            )

        for track in lost_tracks:

            self.track_manager.mark_missed(
                track.id
            )

        for detection in new_detections:

            track = self.track_manager.create_track(
                detection
            )

            self.graph.add_node(

                track.id,

                detection["t"],

                detection["z"],

                detection["y"],

                detection["x"]

            )

    def tracks(self):

        return self.track_manager.all_tracks()

    def graph_nodes(self):

        return self.graph.nodes()

    def graph_edges(self):

        return self.graph.edges()