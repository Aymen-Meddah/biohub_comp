from src.tracking.graph_builder import GraphBuilder
from src.tracking.hungarian_matcher import HungarianMatcher


class Tracker:

    def __init__(self):

        self.graph = GraphBuilder()

        self.matcher = HungarianMatcher()

    def initialize(

        self,

        cells

    ):

        for cell in cells:

            self.graph.add_node(

                cell["id"],

                cell["t"],

                cell["z"],

                cell["y"],

                cell["x"]

            )

    def update(

        self,

        previous_cells,

        current_cells

    ):

        matches = self.matcher.match(

            previous_cells,

            current_cells

        )

        for cell in current_cells:

            self.graph.add_node(

                cell["id"],

                cell["t"],

                cell["z"],

                cell["y"],

                cell["x"]

            )

        for match in matches:

            self.graph.add_edge(

                match["source"]["id"],

                match["target"]["id"]

            )

        return matches

    def graph_nodes(self):

        return self.graph.nodes()

    def graph_edges(self):

        return self.graph.edges()

    def graph_object(self):

        return self.graph.graph