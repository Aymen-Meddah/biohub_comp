import random 
class PatchSampler :
    def __init__(self , patch_size=(32,96,96)):
        self.patch_size = patch_size
    def random_patch(self , volume_shape):
        T, Z, Y, X = volume_shape

        pz, py, px = self.patch_size

        t = random.randint(0, T - 1)

        z0 = random.randint(0, Z - pz)
        y0 = random.randint(0, Y - py)
        x0 = random.randint(0, X - px)

        return {

            "t": t,

            "z0": z0,
            "z1": z0 + pz,

            "y0": y0,
            "y1": y0 + py,

            "x0": x0,
            "x1": x0 + px

        }
    def centered_patch(self ,volume_shape , center):
        T, Z, Y, X = volume_shape

        pz, py, px = self.patch_size

        t, z, y, x = center

        z0 = max(0, z - pz // 2)
        y0 = max(0, y - py // 2)
        x0 = max(0, x - px // 2)

        z0 = min(z0, Z - pz)
        y0 = min(y0, Y - py)
        x0 = min(x0, X - px)

        return {

            "t": t,

            "z0": z0,
            "z1": z0 + pz,

            "y0": y0,
            "y1": y0 + py,

            "x0": x0,
            "x1": x0 + px

        }

    def extract(self ,reader ,patch):
        return reader.patch(

            t=patch["t"],

            z0=patch["z0"],
            z1=patch["z1"],

            y0=patch["y0"],
            y1=patch["y1"],

            x0=patch["x0"],
            x1=patch["x1"]

        )