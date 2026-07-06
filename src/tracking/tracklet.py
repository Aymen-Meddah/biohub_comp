class Tracklet:

    def __init__(

        self,

        tracklet_id

    ):

        self.id = tracklet_id

        self.nodes = []

        self.parent = None

        self.children = []

        self.active = True

    def add_node(

        self,

        node

    ):

        self.nodes.append(node)

    @property
    def first_node(self):

        return self.nodes[0]

    @property
    def last_node(self):

        return self.nodes[-1]

    @property
    def length(self):

        return len(self.nodes)

    def set_parent(

        self,

        parent

    ):

        self.parent = parent

    def add_child(

        self,

        child

    ):

        self.children.append(child)

    def close(self):

        self.active = False

    def to_dict(self):

        return {

            "id": self.id,

            "length": self.length,

            "active": self.active,

            "parent": self.parent,

            "children": self.children

        }