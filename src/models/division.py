class DivisionDetector:
    def __init__(self):
        pass 

    def find_divisions(self ,graph):
        divisions = []
        for node in graph.nodes():
            children = list(graph.successors(node))
            if len(children) >= 2 :
                divisions.append({

                    "parent": node,

                    "children": children

                })
        return divisions
            