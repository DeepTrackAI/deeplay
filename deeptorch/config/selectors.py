class Selector:
      
    def __add__(self, other):
        return ParentalRelation(self, other)
    
    def regex(self, length=None):
        exp = self._regex(length)
        return f"^{exp}$"


class ClassSelector(Selector):

    def __init__(self, className: str):
        self.classes = className.split(" ")

    def __str__(self) -> str:
        if len(self.classes) == 1:
            return self.classes[0]
        
        return f"({' '.join(self.classes)})"
    
    def _regex(self, length=None):
        classes = [c + "(\\[\\d+\\])??" for c in self.classes]
        return f"({'|'.join(classes)})"
    
    def __iter__(self):
        yield from self.classes

class WildcardSelector(Selector):

    def __str__(self) -> str:
        return "*"
    
    def _regex(self, length=None):
        return "[a-zA-Z0-9_]+"
    
    def __iter__(self):
        raise NotImplementedError("WildcardSelector is not iterable")

class DoubleWildcardSelector(Selector):

    def __str__(self) -> str:
        return "**"
    
    def _regex(self, length=None):
        return ".*"
    
    def __iter__(self):
        raise NotImplementedError("DoubleWildcardSelector is not iterable")
    

class IndexSelector(Selector):

    def __init__(self, selector: Selector, index: int or slice):
        self.selector = selector
        self.index = index


    def _get_list_of_indices(self, length):
        if isinstance(self.index, int):
            return [self.index]
        
        stop = length or self.index.stop 


        if stop is None:
            raise TypeError(f"IndexSelector with slice {self.index} must have a stop value")
    
        if stop < 0 and length is None:
            raise TypeError(f"IndexSelector with slice {self.index} must have a length value to support negative indexing")

        if stop < 0:
            stop = stop % length

            print(stop, self.index)

        integers = [i for i in range(stop)][self.index]

        return integers

    def _regex(self, length=None):
        # length is the number of elements in the list being indexed

        integers = self._get_list_of_indices(length)
        integers = "|".join([str(i) for i in integers])
        return self.selector._regex(length) + f"\\[({integers})\\]"

    def __str__(self) -> str:
        return f"{self.selector}[{self.index}]"

    def __iter__(self):
        for selector in self.selector:
            for i in self._get_list_of_indices(None):
                yield f"{selector}[{i}]"


    
      
class ParentalRelation(Selector):

    def __init__(self, parent: Selector, child: Selector):

        self.parent = parent
        self.child = child

    def __str__(self) -> str:
        return f"{self.parent}.{self.child}"
    
    def _regex(self, length=None):

        if isinstance(self.parent, DoubleWildcardSelector) or isinstance(self.child, DoubleWildcardSelector):
            # If either the parent or child is a double wildcard, then the dot separator is optional.
            # This is to allow for edge cases like **.foo to match foo where no dot is present.
            return f"{self.parent._regex(length)}\\.?{self.child._regex(length)}"

        return f"{self.parent._regex(length)}\\.{self.child._regex(length)}"
    
    def __iter__(self):
        for parent in self.parent:
            for child in self.child:
                yield f"{parent}.{child}"