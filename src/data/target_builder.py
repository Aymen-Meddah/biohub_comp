import numpy as np 
class TargetBuilder :
    def __init__(self):
        pass
    def cells_in_patch(self ,nodes ,patch):
        cells = []
        for node_id , data in nodes :
            z = data["z"]
            y = data["y"]
            x = data["x"]
            if (
                patch["z0"] <= z < patch["z1"] and
                patch["y0"] <= y < patch["y1"] and
                patch["x0"] <= x < patch["x1"]
            ):
                cells.append({
                    "id": node_id,

                    "z": z - patch["z0"],

                    "y": y - patch["y0"],

                    "x": x - patch["x0"]
                })
        return cells
    def count_cells(self , nodes ,patch):
        return len(self.cells_in_patch(nodes , patch))
    def has_cells(self , nodes ,patch):
        return self.count_cells(nodes ,patch) > 0
    