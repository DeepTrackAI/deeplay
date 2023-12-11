from .. import DeeplayModule
from ..decorators import before_build


class DeeplayComponent(DeeplayModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._set_last_layer_numeric()

    @before_build
    def _set_last_layer_numeric(self):
        block_name = self.blocks[-1].order[-1]
        getattr(self.blocks[-1], block_name).set_output_map()
