import torch


class ToDict(dict):
    def to(self, device):
        """
        Move all tensors in the dictionary to the specified device.

        Args:
            device (torch.device): The target device.

        Returns:
            ToDict: A new ToDict object with tensors moved to the specified device.
        """
        return ToDict(
            zip(
                self.keys(),
                map(
                    lambda value: value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value,
                    self.values(),
                ),
            )
        )
