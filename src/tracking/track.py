from collections import deque


class Track:

    def __init__(

        self,

        track_id,

        first_cell=None

    ):

        self.id = track_id

        self.active = True

        self.age = 1

        self.missed = 0

        self.parent = None

        self.children = []

        self.history = deque()

        self.nodes = []

        self.kalman = None

        self.missing = 0

        if first_cell is not None:
            self.add_node(
                first_cell
            )

    def add_detection(

        self,

        cell

    ):

        self.history.append(cell)

        self.last_detection = cell

        if hasattr(cell, "track_id"):
            cell.track_id = self.id

        self.age += 1

        self.missed = 0

        self.missing = 0

    def add_node(
        self,
        node
    ):

        self.nodes.append(node)

        self.add_detection(node)

    def mark_missed(self):

        self.missed += 1

    def deactivate(self):

        self.active = False

    def set_parent(

        self,

        parent_track

    ):

        self.parent = parent_track

    def add_child(

        self,

        child_track

    ):

        self.children.append(
            child_track
        )

    @property
    def t(self):

        return self._value("t")

    @property
    def z(self):

        return self._value("z")

    @property
    def y(self):

        return self._value("y")

    @property
    def x(self):

        return self._value("x")

    @property
    def embedding(self):

        if isinstance(self.last_detection, dict):
            return self.last_detection.get(
                "embedding",
                None
            )
        return getattr(self.last_detection, "embedding", None)

    @property
    def position(self):
        return (
            self.z,
            self.y,
            self.x
        )

    def _value(self, key):
        if isinstance(self.last_detection, dict):
            return self.last_detection[key]
        return getattr(self.last_detection, key)

    def to_dict(self):

        return {

            "id": self.id,

            "active": self.active,

            "age": self.age,

            "missed": self.missed,

            "parent": self.parent,

            "children": self.children,

            "history_length": len(
                self.history
            )

        }
