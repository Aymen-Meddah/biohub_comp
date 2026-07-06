import numpy as np


class MotionModel:

    def __init__(self):

        pass

    def predict(
        self,
        track
    ):

        history = list(track.history)

        if len(history) == 1:

            return {

                "z": history[-1]["z"],

                "y": history[-1]["y"],

                "x": history[-1]["x"]

            }

        last = history[-1]

        prev = history[-2]

        vz = last["z"] - prev["z"]

        vy = last["y"] - prev["y"]

        vx = last["x"] - prev["x"]

        return {

            "z": last["z"] + vz,

            "y": last["y"] + vy,

            "x": last["x"] + vx

        }