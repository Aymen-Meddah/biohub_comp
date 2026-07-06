class DetectionMetrics:

    def __init__(self):
        pass

    def compute(

        self,

        matches,

        false_positive,

        false_negative

    ):

        tp = len(matches)

        fp = len(false_positive)

        fn = len(false_negative)

        precision = 0.0

        recall = 0.0

        f1 = 0.0

        if tp + fp > 0:

            precision = tp / (tp + fp)

        if tp + fn > 0:

            recall = tp / (tp + fn)

        if precision + recall > 0:

            f1 = (

                2 *

                precision *

                recall /

                (

                    precision +

                    recall

                )

            )

        return {

            "tp": tp,

            "fp": fp,

            "fn": fn,

            "precision": precision,

            "recall": recall,

            "f1": f1

        }