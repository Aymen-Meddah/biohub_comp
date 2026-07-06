from collections import deque


class Track:

    def __init__(

        self,

        track_id,

        first_cell

    ):

        self.id = track_id

        self.active = True

        self.age = 1

        self.missed = 0

        self.parent = None

        self.children = []

        self.history = deque()

        self.add_detection(
            first_cell
        )

    def add_detection(

        self,

        cell

    ):

        self.history.append(cell)

        self.last_detection = cell

        self.age += 1

        self.missed = 0

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

        return self.last_detection["t"]

    @property
    def z(self):

        return self.last_detection["z"]

    @property
    def y(self):

        return self.last_detection["y"]

    @property
    def x(self):

        return self.last_detection["x"]

    @property
    def embedding(self):

        return self.last_detection.get(
            "embedding",
            None
        )

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