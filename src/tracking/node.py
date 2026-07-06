class Node:

    def __init__(

        self,

        node_id,

        dataset,

        t,

        z,

        y,

        x,

        confidence=1.0,

        embedding=None,

        division_probability=0.0

    ):

        self.id = node_id

        self.dataset = dataset

        self.t = t

        self.z = z

        self.y = y

        self.x = x

        self.confidence = confidence

        self.embedding = embedding

        self.division_probability = division_probability

        self.track_id = None

    @property
    def position(self):

        return (

            self.z,

            self.y,

            self.x

        )

    def to_submission(self):

        return {

            "row_type": "node",

            "node_id": self.id,

            "t": self.t,

            "z": int(round(self.z)),

            "y": int(round(self.y)),

            "x": int(round(self.x)),

            "source_id": -1,

            "target_id": -1

        }

    def to_dict(self):

        return {

            "id": self.id,

            "dataset": self.dataset,

            "t": self.t,

            "z": self.z,

            "y": self.y,

            "x": self.x,

            "confidence": self.confidence,

            "track_id": self.track_id,

            "division_probability": self.division_probability

        }