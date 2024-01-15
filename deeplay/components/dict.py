from typing import Dict, Any, Union, Tuple, overload

from deeplay import DeeplayModule


class FromDict(DeeplayModule):
    def __init__(self, *keys: str):
        super().__init__()
        self.keys = keys

    def forward(self, x: Dict[str, Any]) -> Union[Any, Tuple[Any, ...]]:
        return (
            x[self.keys[0]]
            if len(self.keys) == 1
            else tuple(x[key] for key in self.keys)
        )
