# %%
# To create an __init__ method that accepts both positional and keyword arguments like a dataclass,
# we can use the signature from the class annotations. Here is an example implementation:
import deeplay as dl


model = dl.MultiLayerPerceptron(2, [4], 4)
model.build()
