# Deeplay Models

Models are broadly defined as classes that represent a specific architecture, such as a ResNet18. Unlike `components`, they are
generally not as flexible from input arguments, and it should be possible to pass them directly to applications. Models are designed to be
easy to use and require minimal configuration to get started. They are also designed to be easily extensible, so that you can add new
features without having to modify the existing code.

## What's in a model?

Generally, a model should define a `__init__` method that takes all the necessary arguments to define the model and a `forward` method that
defines the forward pass of the model.

Optimally, a model should have an as simple forward pass as possible. A fully sequential forward pass is optimal.
This is because any hard coded structure in the forward pass limits the flexibility of the model. For example, if the forward pass is defined as
`self.conv1(x) + self.conv2(x)`, then it is not possible to replace `self.conv1` and `self.conv2` with a single `self.conv` without modifying the
model.

Moreover, the model architecture should in almost all cases be defined purely out of components and operations. Try to limit direct calls to
`torch.nn` modules and `blocks`. This is because the `torch.nn` modules are not as flexible as the components and operations in Deeplay. If
components do not exist for the desired architecture, then it is a good idea to create a new component and add it to the `components` folder.
