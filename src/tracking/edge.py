class Edge:

    def __init__(

        self,

        source,

        target

    ):

        self.source = source

        self.target = target

    def to_submission(self):

        return {

            "row_type": "edge",

            "node_id": -1,

            "t": -1,

            "z": -1,

            "y": -1,

            "x": -1,

            "source_id": self.source.id,

            "target_id": self.target.id

        }

    def to_dict(self):

        return {

            "source": self.source.id,

            "target": self.target.id

        }