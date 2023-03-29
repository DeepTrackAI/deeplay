from .selection import Query
from ..parser.parsequery import parse_query
import pytorch_lightning as pl
from typing import List


class Template(pl.LightningModule):
    """Base element for all templates"""

    def __init__(self, children=[], class_name: str = "", id: str = None, **kwargs):

        self.classed(class_name)
        self.id(id)

        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    # ========================================================================
    #                           CLASSNAME METHODS
    # ========================================================================
    def classed(self, class_name: str) -> "Template":
        """Returns a new element with the class added."""
        for class_name in self._get_class_names_from_string(class_name):
            self._add_class(class_name)
        return self

    def has_class(self, class_name: str) -> bool:
        """Returns True if the element has the class."""
        return class_name in self._class_names

    def remove_class(self, class_name: str) -> "Template":
        """Returns a new element with the class removed."""
        if self.has_class(class_name):
            self._class_names.remove(class_name)
        return self

    def toggle_class(self, class_name: str) -> "Template":
        """Returns a new element with the class toggled."""
        if self.has_class(class_name):
            self.remove_class(class_name)
        else:
            self.add_class(class_name)
        return self

    def _add_class(self, class_name: str):
        """Adds a class to the element."""
        if class_name not in self._class_names:
            if not self.has_class(class_name):
                self._class_names.append(class_name)

    @staticmethod
    def _get_class_names_from_string(class_names: str) -> List[str]:
        """Returns a list of class names from a string."""
        return class_names.split()

    # ========================================================================
    #                           ID METHODS
    # ========================================================================

    def id(self, id: str) -> "Template":
        """Returns a new element with the id set."""
        self._id = id
        return self

    def has_id(self, id: str) -> bool:
        """Returns True if the element has the id."""
        return self._id == id

    def remove_id(self) -> "Template":
        """Returns a new element with the id removed."""
        self._id = None
        return self

    def toggle_id(self, id: str) -> "Template":
        """Returns a new element with the id toggled."""
        if self.has_id(id):
            self.remove_id()
        else:
            self.id(id)
        return self

    # =======================================================================
    #                           TAG METHODS
    # =======================================================================

    def get_tag(self) -> str:
        """Returns the tag of the element."""
        return self.__name__

    def has_tag(self, tag: str) -> bool:
        """Returns True if the element has the tag."""
        return self.get_tag() == tag

    # =======================================================================
    #                           MATCH METHODS
    # =======================================================================

    def matches(self, query: Query) -> bool:
        query = parse_query(query)
        return query.matches(self)

    def select(self, query: Query) -> List["Template"]:
        query = parse_query(query)
        return query.select(self)

    def select_all(self, query: Query) -> List["Template"]:
        query = parse_query(query)
        return query.select_all(self)
