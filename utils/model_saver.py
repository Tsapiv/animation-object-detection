import os

import torch


class ModelSaver:
    def __init__(self, root, topk: int = 3, minimize: bool = True):
        self.topk = topk
        self.root = root
        self.minimize = minimize
        self.rank = []
        os.makedirs(self.root, exist_ok=True)

    def update(self, model, criterion, epoch):
        if len(self.rank) < self.topk:
            self.rank.append((criterion, self.generate_name(criterion, epoch)))
            torch.save(model, self.generate_name(criterion, epoch))
        else:
            self.rank.sort(key=lambda x: x[0] if self.minimize else -x[0])
            if self.rank[-1][0] > criterion:
                _, path = self.rank.pop()
                os.unlink(path)
                self.rank.append((criterion, self.generate_name(criterion, epoch)))
                torch.save(model, self.generate_name(criterion, epoch))

    def generate_name(self, value, epoch):
        return os.path.join(self.root, f"epoch-{epoch}-{value}.pth")
