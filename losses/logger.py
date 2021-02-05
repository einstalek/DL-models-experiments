from collections import defaultdict

class Logger:
    def __init__(self):
        self.loss = defaultdict(list)

    def update(self, **loss):
        for (k, v) in loss.items():
            self.loss[k].append(v)
