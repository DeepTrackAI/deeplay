# Deeplay for developers

**NOTE**: This guideline is _not_ complete or final. If anything is unclear, please open an issue or a pull request to improve it. The goal is to evolve this document to be a comprehensive guide for developers who want to contribute to the Deeplay project by iteratively incorporating feedback from the community.

## Introduction

This document is intended for developers who want to contribute to the Deeplay project. It provides guidelines how to structure the code, how to write tests, how to document the code, and how to contribute to the project. It also helps decide if the features you want to implement are `Application`s or `Model`s or any other type of object. Finally, it outlines how to create and format a pull request.

## Adding a new feature

When adding a new feature, you should first decide what type of object it is.

### New deep learning classes

The following table provides a sequence of questions to help you decide what type of object you are implementing. Each
possible answer will have its own style guide in the folder of the same name.

| Question                                                                                                         | Answer | Object type   |
| ---------------------------------------------------------------------------------------------------------------- | ------ | ------------- |
| Does the class represent a task such as classification, without depending heavily on the exact architecture?     | Yes    | `Application` |
| Does the class require a non-standard training procedure to make sense (standard is normal supervised learning)? | Yes    | `Application` |
| Does the object represent a specific architecture, such as a ResNet18?                                           | Yes    | `Model`       |
| Does the object represent a full architecture, not expected to be used as a part of another architecture?        | Yes    | `Model`       |
| Does the object represent a generic architecture, such as a ConvolutionalNeuralNetwork?                          | Yes    | `Component`   |
| Is the object mostly structural with a sequential forward pass, such as `LayerActivation`?                       | Yes    | `Block`       |
| Is the object a unit of computation, such as `Conv2d`, `Add` etc.?                                               | Yes    | `Operation`   |

## Pull requests

When creating a pull request, you should follow the following steps:

1. Create a fork of the repository.
2. Create a new branch for your feature.
3. Implement the feature.
4. Write tests for the feature.
5. Write documentation for the feature.
6. Create a pull request.

The pull request should contain the following:

- An overview of why the feature was added and when it should be used.
- A description of the feature.
- A description of all new classes and methods, and why they were added. Include why existing classes and methods were not sufficient.

## Code style

The code style should follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines. The code should be formatted using
[black](https://black.readthedocs.io/en/stable/). We are not close to lint-compliance yet, but we are working on it.

Use type hints extensively to make the code more readable and maintainable. The type hints should be as specific as possible.
For example, if a string can be one of several values, use a `Literal` type hint. Similarly, if a function takes a list of integers,
the type hint should be `List[int]` instead of `List`. We are currently supporting Python 3.8 and above. Some features of Python 3.9
are not supported yet, such as the `|` operator for type hints.

Classes should have their attribute types defined before the `__init__` method. An example is shown below:

```python
class MyClass:
    attribute: int

    def __init__(self, attribute: int):
        self.attribute = attribute
```

### Naming conventions

Beyond what is defined in the PEP 8 guidelines, we have the following naming conventions:

- Minimize the use of abbreviations. If an abbreviation is used, it should be well-known and not ambiguous.
- Use the following names:
  - "layer" for a class that represents a single layer in a neural network, typically the learnable part of a block.
  - "activation" for a class that represents a non-learnable activation function.
  - "normalization" for a class that represents a normalization layer.
  - "dropout" for a class that represents a dropout layer.
  - "pool" for a class that represents a pooling layer.
  - "block" / "blocks" for a class that represents a block in a neural network, typically a sequence of layers.
  - "backbone" for a class that represents the main part of a neural network, typically a sequence of blocks.
  - "head" for a class that represents the final part of a neural network, typically a single layer followed by an optional activation function.
  - "model" for a class that represents a full neural network architecture.
  - "optimizer" for a class that represents an optimizer.
  - "loss" for a class that represents a loss function.
  - "metric" for a class that represents a metric.
- If there is a naming conflict within a class, add numbers to the end of the name with an underscore, 0-indexed. For example, `layer_0` and `layer_1`.
  - This is correct: `layer_0`, `layer_1`, `layer_2`.
  - This is incorrect: `layer_1`, `layer_2`, `layer_3`.
  - This is incorrect: `layer`, `layer_1`, `layer_3`.

## Documentation

Documentation should follow the [NumpyDoc style guide](https://numpydoc.readthedocs.io/en/latest/format.html#style-guide).

In general, all non-trivial classes and methods should be documented. The documentation should include a description of the class or method, the parameters, the return value, and any exceptions that can be raised. We sincerely appreciate any effort to improve the documentation, particularly by including examples of how to use the classes and methods.

## Testing

All new features should be tested. The tests should cover all possible code paths and should be as comprehensive as possible. The tests should be written using the `unittest` module in Python. The tests should be placed in the `tests` folder. The tests should be run using `unittest`. Not all tests follow our guidelines yet, but we are working on improving them.

In general, we aim to mirror the structure of the `deeplay` package in the `tests` package. For example, `deeplay.external.layer` should have a corresponding `tests/external/test_layer.py` file. The name of the file should be the same as the module it tests, but with `test_` prepended. Note that when adding a folder, the `__init__.py` file should be added to the folder to make it a package.

Each test file should contain one `unittest.TestCase` class per class or method being tested. The test methods should be as descriptive as possible. For example, `test_forward` is not a good name for a test method. Instead, use `test_forward_with_valid_input` or `test_forward_with_invalid_input`. The test methods should be as independent as possible, but we value coverage over independence.

It is fine to use multiple subtests using `with self.subTest()` to test multiple inputs or edge cases. This is particularly useful when testing a method that has many possible inputs.

It is fine and preferred to use mocking where appropriate. For example, if a method calls an external API, the API call should be mocked. The `unittest.mock` module is very useful for this purpose.

### Testing applications

See the [testing applications](application/testing.md) guidelines.

### Testing models

See the [testing models](model/testing.md) guidelines.
