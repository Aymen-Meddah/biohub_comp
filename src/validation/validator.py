from src.validation.evaluator import Evaluator


class Validator:

    def __init__(

        self,

        model,

        device,

        threshold=0.30,

        max_distance=10.0

    ):

        self.evaluator = Evaluator(

            model=model,

            device=device,

            threshold=threshold,

            max_distance=max_distance

        )

        self.best_score = 0.0

    def validate(

        self,

        dataloader

    ):

        results = self.evaluator.evaluate(

            dataloader

        )

        current_score = float(results["f1"])

        is_best = current_score > self.best_score + 1e-12

        if is_best:

            self.best_score = current_score

        return {

            "metrics": results,

            "best_score": self.best_score,

            "is_best": is_best

        }