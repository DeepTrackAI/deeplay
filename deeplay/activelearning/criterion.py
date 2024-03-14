import torch


class ActiveLearningCriterion:

    def score(self, probabilities):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, ActiveLearningCriterion):
            return SumCriterion(self, other)
        elif isinstance(other, (float, int, bool)):
            return SumCriterion(self, Constant(other))
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, ActiveLearningCriterion):
            return ProductCriterion(self, other)
        elif isinstance(other, (float, int, bool)):
            return ProductCriterion(self, Constant(other))
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, ActiveLearningCriterion):
            return SumCriterion(self, other * -1)
        elif isinstance(other, (float, int, bool)):
            return SumCriterion(self, Constant(-other))
        else:
            raise NotImplementedError

    def __div__(self, other):
        if isinstance(other, ActiveLearningCriterion):
            return FractionCriterion(self, other)
        elif isinstance(other, (float, int, bool)):
            return FractionCriterion(self, other)
        else:
            raise NotImplementedError

    def __rdiv__(self, other):
        if isinstance(other, ActiveLearningCriterion):
            return FractionCriterion(other, self)
        elif isinstance(other, (float, int, bool)):
            return FractionCriterion(Constant(other), self)
        else:
            raise NotImplementedError


class Constant(ActiveLearningCriterion):
    def __init__(self, value):
        self.value = value

    def score(self, probabilities):
        return torch.full((probabilities.shape[0],), self.value).to(
            probabilities.device
        )


class LeastConfidence(ActiveLearningCriterion):
    def score(self, probabilities):
        return torch.max(probabilities, dim=1).values


class Margin(ActiveLearningCriterion):
    def score(self, probabilities):
        sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
        return sorted_probs[:, 0] - sorted_probs[:, 1]


class Entropy(ActiveLearningCriterion):
    def score(self, probabilities):
        return -torch.sum(probabilities * torch.log(probabilities), dim=1)


class L1Upper(ActiveLearningCriterion):
    def score(self, probabilities):
        return torch.log(probabilities).sum(dim=1) * -1


class L2Upper(ActiveLearningCriterion):
    def score(self, probabilities):
        return torch.norm(torch.log(probabilities), dim=1)


class SumCriterion(ActiveLearningCriterion):
    def __init__(self, *criterion):
        self.criterion = criterion

    def score(self, probabilities):
        p = self.criterion[0].score(probabilities)
        for criterion in self.criterion[1:]:
            p += criterion.score(probabilities)
        return p


class ProductCriterion(ActiveLearningCriterion):
    def __init__(self, *criterion):
        self.criterion = criterion

    def score(self, probabilities):
        p = self.criterion[0].score(probabilities)
        for criterion in self.criterion[1:]:
            p *= criterion.score(probabilities)
        return p


class FractionCriterion(ActiveLearningCriterion):
    def __init__(self, *criterion):
        self.criterion_1 = criterion[0]
        self.criterion_2 = criterion[1]

    def score(self, probabilities):
        return self.criterion_1.score(probabilities) / self.criterion_2.score(
            probabilities
        )
