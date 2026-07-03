from pathlib import Path
import geff 
class GeffReader :
        def __init__(self ,geff_path):
            self.geff_path = Path(geff_path)
            if not self.geff_path.exists():
                raise FileNotFoundError(
                    F"DEFF FILE NOT FOUND :{self.geff_path}"
                )
            self.graph , self.metadata = geff.read(self.geff_path)

        @property
        def number_of_nodes(self):
            return self.graph.number_of_nodes()
         
        @property
        def number_of_adges(self):
            return self.graph.number_of_edges()
        
        def nodes(self):
            return list(self.graph.nodes(data=True))
        def edges(self):
            return list(self.graph.edges())
        def node(self, node_id):
            return self.graph.nodes[node_id]
        def nodes_at_time(self , t):
            result = []
            for node_id , data in self.graph.nodes(data=True):
                if data["t"] == t :
                    result.append((node_id ,data))
            return result 
        def successors(self ,node_id):
            return list(self.graph.seccessors( node_id))
        def predecessors(self ,node_id):
            return list(self.graph.predecessors(node_id))
        def didision_nodes(self):
            division = []
            for node in self.graph.node():
                if self.graph.out_degree(node) >= 2 :
                    division.append(node)
            return division
        def summary(self):
            return{

                "nodes": self.number_of_nodes,

                "edges": self.number_of_edges,

                "divisions": len(self.division_nodes())
            }
        def __repr__(self):
            return (
                f"GeffReader("
                f"nodes={self.number_of_nodes}, "
                f"edges={self.number_of_edges})" )
        
