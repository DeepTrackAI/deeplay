Deeplay is a deep learning library in Python that extends PyTorch with additional functionalities focused on modularity and reusability. It facilitates the definition, training, and adjustment of neural networks by introducing dynamic modification capabilities for model components after their initial creation. Deeplay seeks to address the common issue of rigid and non-reusable modules in PyTorch projects by offering a system that allows for easy customization and optimization of neural network components.

# Core Philosophy

The core philosophy of Deeplay is to enhance flexibility in the construction and adaptation of neural networks. It is built on the observation that PyTorch modules often lack reusability across projects, leading to redundant implementations. Deeplay enables properties of neural network submodules to be changed post-creation, supporting seamless integration of these modifications. Its design is based on a hierarchy of abstractions from models down to layers, emphasizing compatibility and easy transformation of components. This can be summarized aqs follows:

- **Enhance Flexibility:** Neural networks defined using Deeplay should be fully adaptable by the user, allowing dynamic modifications to model components. This should be possible without the author of the model having to anticipate all potential changes in advance.
- **Promote Reusability:** Deeplay components should be immediately reusable across different projects and models. This reusability should extend to both the components themselves and the modifications made to them.
- **Support Seamless Integration:** Modifications to model blocks and components should be possible without the user worrying about breaking the model's compatibility with other parts of the network. Deeplay should handle these integrations automatically as far as possible.
- **Hierarchy of Abstractions:** Neural networks and deep learning are fundamentally hierarchical, with each level of abstraction being mostly agnostic to the details of the levels below it. An application should be agnostic to which model it uses, a model should be agnostic to the specifics of the components it uses, a component should be agnostic to the specifics of the blocks it uses, and so on. Deeplay reflects this hierarchy in its design.

# Deeplay Compared to Torch

Deeplay is designed as a superset of PyTorch, retaining compatibility with PyTorch code while introducing features aimed at improving modularity and customization. Unlike PyTorch's fixed module implementations, Deeplay provides a framework that supports dynamic adjustments to model architectures. This includes capabilities for on-the-fly property changes and a style registry for component customization. Users can easily transition between PyTorch and Deeplay, taking advantage of Deeplay's additional features without losing the familiarity and functionality of PyTorch.

# Deeplay Compared to Lightning

While Deeplay utilizes PyTorch Lightning for simplifying the training loop process, it goes further by offering enhanced modularity for the architectural design of models. PyTorch Lightning focuses on streamlining and optimizing training operations, whereas Deeplay extends this convenience to the model construction phase. This integration offers users a comprehensive toolset for both designing flexible neural network architectures and efficiently managing their training, positioning Deeplay as a solution for more adaptive and intuitive neural network development.

# Quick Start Guide

The following quick start guide is intended for complete beginners to understand how to use Deeplay, from installation to training your first model. Let's get started!

## Installation

You can install Deeplay using pip:
```bash
pip install deeplay
```
or
```bash
python -m pip install deeplay
```
This will automatically install the required dependencies, including PyTorch and PyTorch Lightning. If a specific version of PyTorch is desired, it can be installed separately.

## Getting Started

Here you find a series of notebooks that give you an overview of the core features of Deeplay and how to use them:

- GS101 **[Understanding the Core Objects in Deeplay](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS101_core_objects.ipynb)**

  Layers, Blocks, Components, Models, Applications.

- GS111 **[Training Your First Model](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS111_first_model.ipynb)**

  Creating, training, saving and using a deep learning model with Deeplay.

- GS121 **[Working with Deeplay Modules](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS121_modules.ipynb)**

  Differences between Deeplay and PyTorch modules. How to create, build, and configure Deeplay modules.

- GS131 **[Using the Main Deeplay Methods](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS131_methods.ipynb)**

  `Application.fit()`, `Application.test()`, `DeeplayModule.predict()`, `Trainer.fit()`.

- GS141 **[Using Deeplay Applications](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS141_applications.ipynb)**

  Main Deeplay applications. Controlling loss functions, optimizers, and metrics. Training history. Callback.

- GS151 **[Using Deeplay Models](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS151_models.ipynb)**

  Main Deeplay models. Making a model. Weight initialization.

- GS161 **[Using Deeplay Components](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS161_components.ipynb)**

  Main Deeplay components.

- GS171 **[Using Deeplay Blocks](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS171_blocks.ipynb)**

  Main Deeplay blocks. Adding, ordering, and removing layers. Operations.

- GC181 **[Configuring Deeplay Objects](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS181_configure.ipynb)**

  `DeeplayModule.configure()` and selectors.

- GC191 **[Using Styles](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS191_styles.ipynb)**

  Styles.

## Getting Started

Here you find a series of notebooks tailored for Deeplay's developers:

