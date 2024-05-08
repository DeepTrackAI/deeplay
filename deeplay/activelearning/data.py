import numpy as np
import torch
import torch.utils.data


class ActiveLearningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.annotated = np.zeros(len(dataset), dtype=bool)

    def __getitem__(self, index):
        return self.dataset[index], self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def annotate_random(self, n) -> np.ndarray:
        """Annotate n random samples."""
        indices = np.random.choice(len(self.dataset), n, replace=False)
        self.annotated[indices] = True
        return indices

    def annotate(self, indices: np.ndarray):
        """Annotate specific samples."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        unannotated_indices = np.where(~self.annotated)[0]
        self.annotated[unannotated_indices[indices]] = True

    def get_annotated_samples(self):
        data = self.get_annotated_data()
        X = [x for x, *_ in data]
        return torch.stack(X)

    def get_annotated_labels(self):
        data = self.get_annotated_data()
        Y = [y for _, y in data]
        return torch.stack(Y)

    def get_annotated_data(self):
        return torch.utils.data.Subset(self.dataset, np.where(self.annotated)[0])

    def get_unannotated_samples(self):
        data = self.get_unannotated_data()
        X = [x for x, *_ in data]
        return torch.stack(X)

    def get_unannotated_labels(self):
        data = self.get_unannotated_data()
        Y = [y for _, y in data]
        try:
            return torch.stack(Y)
        except TypeError:
            return torch.LongTensor(Y)

    def get_unannotated_data(self):
        return torch.utils.data.Subset(self.dataset, np.where(~self.annotated)[0])

    def get_num_annotated(self):
        return np.sum(self.annotated)


class JointDataset(torch.utils.data.Dataset):

    def __init__(self, X_1, Y_1, X_2, Y_2):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1), len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        return index, x_1, y_1, x_2, y_2
