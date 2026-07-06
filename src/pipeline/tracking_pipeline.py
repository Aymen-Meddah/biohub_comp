from src.inference.decoder import CellDecoder
from src.tracking.hungarian_tracker import HungarianTracker
from src.tracking.track_manager import TrackManager


class TrackingPipeline:

    def __init__(

        self,

        model,

        device,

        threshold=0.30,

        max_cost=30.0

    ):

        self.model = model

        self.device = device

        self.decoder = CellDecoder(

            threshold=threshold

        )

        self.tracker = HungarianTracker(

            max_cost=max_cost

        )

        self.manager = TrackManager()

    def process_frame(

        self,

        image,

        dataset,

        timepoint

    ):

        self.model.eval()

        outputs = self.model(

            image

        )

        nodes = self.decoder.decode(

            outputs,

            dataset,

            timepoint

        )

        if len(self.manager.tracks) == 0:

            for node in nodes:

                self.manager.create_track(

                    node

                )

            return self.manager.tracks

        self.manager.predict_tracks()

        previous_nodes = [

            track.nodes[-1]

            for track in self.manager.tracks

        ]

        matches, lost, new = self.tracker.associate(

            previous_nodes,

            nodes

        )

        for previous_node, current_node, _ in matches:

            for track in self.manager.tracks:

                if track.nodes[-1] == previous_node:

                    self.manager.update_track(

                        track,

                        current_node

                    )

                    break

        for node in new:

            self.manager.create_track(

                node

            )

        self.manager.remove_dead_tracks()

        return self.manager.tracks