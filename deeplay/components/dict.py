from typing import Dict, Any, Union, Tuple
from deeplay import DeeplayModule


class FromDict(DeeplayModule):
    def __init__(self, *names: str):
        super().__init__()
        self.names = names

    def forward(self, x: Dict[str, Any]) -> Union[Any, Tuple[Any, ...]]:
        return (
            x[self.names[0]]
            if len(self.names) == 1
            else tuple(x[name] for name in self.names)
        )
