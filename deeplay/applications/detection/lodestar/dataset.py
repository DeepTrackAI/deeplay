import torch.utils.data


class LodeSTARDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, transform: Callable[[torch.Tensor], torch.Tensor]):
        self._path = path
        self._transform = transform

    def __len__(self) -> int:
        return len(self._path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self._path[index]
        image = torch.load(image_path)
        image = self._transform(image)
        return image, image_path

class 