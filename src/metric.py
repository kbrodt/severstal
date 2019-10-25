def calc_dice(pred, target, thresh=0.5, eps=1e-8, min_area=float('-inf')):
    pred = pred > thresh
    if pred.sum() < min_area:
        pred[:] = False

    pred = pred.flatten()
    target = target.flatten()

    inter = (pred * target).sum()
    union = pred.sum() + target.sum()

    return (2 * inter + eps) / (union + eps)


class Dice:
    def __init__(self, n_classes=1, thresh=None):
        self.thresh = thresh
        self.n_classes = n_classes
        self.clean()

    def clean(self):
        self.dice = {i: 0.0 for i in range(self.n_classes)}
        self.n = 0

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        for p, t in zip(preds, targets):
            if self.thresh is None:
                p = p.argmax(0)
                for c in range(self.n_classes):
                    self.dice[c] += calc_dice(p == c - 1, t == c - 1)
            else:
                for c in range(self.n_classes):
                    self.dice[c] += calc_dice(p[c], t[c], self.thresh)
        self.n += len(preds)

    def evaluate(self, reduce=True):
        if not reduce:
            return [self.dice[c]/self.n for c in self.dice]

        if self.n > 0:
            return sum(self.dice.values())/self.n_classes/self.n

        return 0.0

    
class Accuracy:
    def __init__(self, thresh, n_classes=1):
        self.thresh = thresh
        self.n_classes = n_classes
        self.clean()

    def clean(self):
        self.acc = {i: 0.0 for i in range(self.n_classes)}
        self.n = 0

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        for p, t in zip(preds, targets):
            t = t.sum((1,2)) > 0
            for c in range(self.n_classes):
                self.acc[c] += t[c] == (p[c] > self.thresh)
        self.n += len(preds)

    def evaluate(self, reduce=True):
        if not reduce:
            return [self.acc[c]/self.n for c in self.acc]

        if self.n > 0:
            return sum(self.acc.values())/self.n_classes/self.n

        return 0.0
