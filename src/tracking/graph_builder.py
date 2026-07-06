import networkx as nx


class GraphBuilder:

    def __init__(self):

        self.graph = nx.DiGraph()

    def add_node(
        self,
        node_id,
        t,
        z,
        y,
        x
    ):

        self.graph.add_node(

            node_id,

            t=t,

            z=z,

            y=y,

            x=x

        )

    def add_edge(
        self,
        source,
        target
    ):

        self.graph.add_edge(
            source,
            target
        )

    def remove_node(
        self,
        node_id
    ):

        if node_id in self.graph:

            self.graph.remove_node(
                node_id
            )

    def remove_edge(
        self,
        source,
        target
    ):

        if self.graph.has_edge(
            source,
            target
        ):

            self.graph.remove_edge(
                source,
                target
            )

    def nodes(self):

        return list(
            self.graph.nodes(data=True)
        )

    def edges(self):

        return list(
            self.graph.edges()
        )

    def number_of_nodes(self):

        return self.graph.number_of_nodes()

    def number_of_edges(self):

        return self.graph.number_of_edges()

    def clear(self):

        self.graph.clear()