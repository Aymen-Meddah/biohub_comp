class AverageMeter:

    def __init__(self):

        self.reset()

    def reset(self):

        self.value = 0.0

        self.sum = 0.0

        self.count = 0

        self.average = 0.0

    def update(

        self,

        value,

        n=1

    ):

        self.value = float(value)

        self.sum += float(value) * n

        self.count += n

        self.average = self.sum / self.count

    @property

    def avg(self):

        return self.average

    def __str__(self):

        return f"{self.average:.6f}"