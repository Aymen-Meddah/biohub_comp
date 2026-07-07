from pathlib import Path


class GeffReader:

    TIME_KEYS = ("t", "time", "timepoint", "frame")
    Z_KEYS = ("z", "Z", "centroid_z", "position_z")
    Y_KEYS = ("y", "Y", "centroid_y", "position_y")
    X_KEYS = ("x", "X", "centroid_x", "position_x")
    RADIUS_KEYS = ("radius", "r", "cell_radius")

    def __init__(self, geff_path):
        self.geff_path = Path(geff_path)
        if not self.geff_path.exists():
            raise FileNotFoundError(
                f"GEFF file not found: {self.geff_path}"
            )

        try:
            import geff
        except ImportError as exc:
            raise ImportError(
                "The 'geff' package is required to read BioHub metadata."
            ) from exc

        self.graph, self.metadata = geff.read(self.geff_path)

    @property
    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def number_of_edges(self):
        return self.graph.number_of_edges()

    def nodes(self):
        return list(self.graph.nodes(data=True))

    def edges(self):
        return list(self.graph.edges())

    def node(self, node_id):
        return self.graph.nodes[node_id]

    @staticmethod
    def _first_value(data, keys, default=None):
        for key in keys:
            if key in data:
                return data[key]
        return default

    def normalize_node(self, node_id, data):
        normalized = dict(data)
        normalized["id"] = node_id
        normalized["t"] = int(self._first_value(data, self.TIME_KEYS, 0))
        normalized["z"] = float(self._first_value(data, self.Z_KEYS, 0.0))
        normalized["y"] = float(self._first_value(data, self.Y_KEYS, 0.0))
        normalized["x"] = float(self._first_value(data, self.X_KEYS, 0.0))

        radius = self._first_value(data, self.RADIUS_KEYS)
        if radius is not None:
            normalized["radius"] = float(radius)

        normalized["division"] = self.graph.out_degree(node_id) >= 2
        return normalized

    def nodes_at_time(self, t):
        result = []
        for node_id, data in self.graph.nodes(data=True):
            node = self.normalize_node(node_id, data)
            if node["t"] == int(t):
                result.append((node_id, node))
        return result

    def successors(self, node_id):
        return list(self.graph.successors(node_id))

    def predecessors(self, node_id):
        return list(self.graph.predecessors(node_id))

    def division_nodes(self):
        return [
            node
            for node in self.graph.nodes()
            if self.graph.out_degree(node) >= 2
        ]

    def summary(self):
        return {
            "nodes": self.number_of_nodes,
            "edges": self.number_of_edges,
            "divisions": len(self.division_nodes())
        }

    def __repr__(self):
        return (
            f"GeffReader("
            f"nodes={self.number_of_nodes}, "
            f"edges={self.number_of_edges})"
        )


GEFFReader = GeffReader
