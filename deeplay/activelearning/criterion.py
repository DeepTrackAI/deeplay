import torch


class ActiveLearningCriteria:

    def score(self, probabilities):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, ActiveLearningCriteria):
            return SumCriterion(self, other)
        elif isinstance(other, (float, int, bool)):
            return SumCriterion(self, Constant(other))
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, ActiveLearningCriteria):
            return ProductCriterion(self, other)
        elif isinstance(other, (float, int, bool)):
            return ProductCriterion(self, Constant(other))
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, ActiveLearningCriteria):
            return SumCriterion(self, other * -1)
        elif isinstance(other, (float, int, bool)):
            return SumCriterion(self, Constant(-other))
        else:
            raise NotImplementedError

    def __div__(self, other):
        if isinstance(other, ActiveLearningCriteria):
            return FractionCriterion(self, other)
        elif isinstance(other, (float, int, bool)):
            return FractionCriterion(self, other)
        else:
            raise NotImplementedError

    def __rdiv__(self, other):
        if isinstance(other, ActiveLearningCriteria):
            return FractionCriterion(other, self)
        elif isinstance(other, (float, int, bool)):
            return FractionCriterion(Constant(other), self)
        else:
            raise NotImplementedError


class Constant(ActiveLearningCriteria):
    def __init__(self, value):
        self.value = value

    def score(self, probabilities):
        return torch.full((probabilities.shape[0],), self.value).to(probabilities.device)


class LeastConfidence(ActiveLearningCriteria):
    def score(self, probabilities):
        return torch.max(probabilities, dim=1).values


class Margin(ActiveLearningCriteria):
    def score(self, probabilities):
        sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
        return sorted_probs[:, 0] - sorted_probs[:, 1]


class Entropy(ActiveLearningCriteria):
    def score(self, probabilities):
        return -torch.sum(probabilities * torch.log(probabilities), dim=1)


class L1Upper(ActiveLearningCriteria):
    def score(self, probabilities):
        return torch.log(probabilities).sum(dim=1) * -1


class L2Upper(ActiveLearningCriteria):
    def score(self, probabilities):
        return torch.norm(torch.log(probabilities), dim=1)


class SumCriterion(ActiveLearningCriteria):
    def __init__(self, *criterion):
        self.criterion = criterion

    def score(self, probabilities):
        p = self.criterion[0].score(probabilities)
        for criterion in self.criterion[1:]:
            p += criterion.score(probabilities)
        return p


class ProductCriterion(ActiveLearningCriteria):
    def __init__(self, *criterion):
        self.criterion = criterion

    def score(self, probabilities):
        p = self.criterion[0].score(probabilities)
        for criterion in self.criterion[1:]:
            p *= criterion.score(probabilities)
        return p


class FractionCriterion(ActiveLearningCriteria):
    def __init__(self, *criterion):
        self.criteria_1 = criterion[0]
        self.criteria_2 = criterion[1]

    def score(self, probabilities):
        return self.criteria_1.score(probabilities) / self.criteria_2.score(
            probabilities
        )
