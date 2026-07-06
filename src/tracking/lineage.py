from src.tracking.node import Node
from src.tracking.edge import Edge


class Lineage:

    def __init__(self):

        self.nodes = {}

        self.edges = []

    def add_node(

        self,

        node

    ):

        self.nodes[node.id] = node

    def add_edge(

        self,

        source,

        target

    ):

        edge = Edge(

            source,

            target

        )

        self.edges.append(edge)

    def get_node(

        self,

        node_id

    ):

        return self.nodes.get(node_id)

    def node_exists(

        self,

        node_id

    ):

        return node_id in self.nodes

    def all_nodes(self):

        return list(

            self.nodes.values()

        )

    def all_edges(self):

        return self.edges

    def number_of_nodes(self):

        return len(

            self.nodes

        )

    def number_of_edges(self):

        return len(

            self.edges

        )

    def clear(self):

        self.nodes.clear()

        self.edges.clear()

    def to_submission_rows(self):

        rows = []

        for node in self.all_nodes():

            rows.append(

                node.to_submission()

            )

        for edge in self.all_edges():

            rows.append(

                edge.to_submission()

            )

        return rows