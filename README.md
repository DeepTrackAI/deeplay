<h3 align="center">Deeplay - A Modular Superset of PyTorch</h3>
<p align="center">
  <a href="/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
  </a>
  <a href="https://badge.fury.io/py/deeplay">
    <img src="https://badge.fury.io/py/deeplay.svg" alt="PyPI version">
  </a>
  <a href="https://deeptrackai.github.io/deeplay">
    <img src="https://img.shields.io/badge/docs-available-blue?logo=readthedocs">
  </a>
  <a href="https://badge.fury.io/py/deeptrack">
    <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue" alt="Python version">
  </a>
</p>

Deeplay is a deep learning library in Python that extends PyTorch with additional functionalities focused on modularity and reusability. Deeplay seeks to address the common issue of rigid and non-reusable modules in PyTorch projects by offering a system that allows for easy customization and optimization of neural network components. Specifically, it facilitates the definition, training, and adjustment of neural networks by introducing dynamic modification capabilities for model components after their initial creation.

# Core Philosophy

The core philosophy of Deeplay is to enhance flexibility in the construction and adaptation of neural networks. It is built on the observation that PyTorch modules often lack reusability across projects, leading to redundant implementations. Deeplay enables properties of neural network submodules to be changed post-creation, supporting seamless integration of these modifications. Its design is based on a hierarchy of abstractions from models down to layers, emphasizing compatibility and easy transformation of components. This can be summarized as follows:

- **Enhance Flexibility:** Neural networks defined using Deeplay should be fully adaptable by the user, allowing dynamic modifications to model components. This should be possible without the author of the model having to anticipate all potential changes in advance.
- **Promote Reusability:** Deeplay components should be immediately reusable across different projects and models. This reusability should extend to both the components themselves and the modifications made to them.
- **Support Seamless Integration:** Modifications to model blocks and components should be possible without the user worrying about breaking the model's compatibility with other parts of the network. Deeplay should handle these integrations automatically as far as possible.
- **Hierarchy of Abstractions:** Neural networks and deep learning are fundamentally hierarchical, with each level of abstraction being mostly agnostic to the details of the levels below it. An *application* should be agnostic to which model it uses, a *model* should be agnostic to the specifics of the components it uses, a *component* should be agnostic to the specifics of the blocks it uses, and a *block* should be agnostic to the specifics of the *layers* it uses . Deeplay reflects this hierarchy in its design.

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

- GS101 **[Understanding the Core Objects in Deeplay](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS101_core_objects.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS101_core_objects.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Layers, Blocks, Components, Models, Applications.

- GS111 **[Training Your First Model](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS111_first_model.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS111_first_model.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Creating, training, saving and using a deep learning model with Deeplay.

- GS121 **[Working with Deeplay Modules](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS121_modules.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS121_modules.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Differences between Deeplay and PyTorch modules. How to create, build, and configure Deeplay modules.

- GS131 **[Using the Main Deeplay Methods](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS131_methods.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS131_methods.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  `Application.fit()`, `Application.test()`, `DeeplayModule.predict()`, `Trainer.fit()`.

- GS141 **[Using Deeplay Applications](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS141_applications.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS141_applications.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Main Deeplay applications. Controlling loss functions, optimizers, and metrics. Training history. Callback.

- GS151 **[Using Deeplay Models](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS151_models.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS151_models.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Main Deeplay models. Making a model. Weight initialization.

- GS161 **[Using Deeplay Components](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS161_components.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS161_components.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Main Deeplay components.

- GS171 **[Using Deeplay Blocks](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS171_blocks.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS171_blocks.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Main Deeplay blocks. Adding, ordering, and removing layers. Operations.

- GC181 **[Configuring Deeplay Objects](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS181_configure.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS181_configure.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  `DeeplayModule.configure()` and selectors.

- GC191 **[Using Styles](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS191_styles.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/getting-started/GS191_styles.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Styles.

## Examples

## Advanced Topics

- AT201 **[Using Mappings as Inputs](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/advanced-topics/AT201_mappings.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/advanced-topics/AT201_mappings.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

## Developer Tutorials

Here you find a series of notebooks tailored for Deeplay's developers:

- DT101 **[Deeplay File Structure](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT101_files.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT101_files.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DT111 **[Style Guide](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT111_style.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT111_style.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DT121 **[Overview of Deeplay Classes](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT121_overview.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT121_overview.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DT131 **[Deeplay Applications](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT131_applications.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT131_applications.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DT141 **[Deeplay Models](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT141_models.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT141_models.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DT151 **[Deeplay Components](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT151_components.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT151_components.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DT161 **[Deeplay Operations](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT151_operations.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT151_operations.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DT171 **[Deeplay Blocks](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT171_vlocks.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT171_vlocks.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DT181 **[Overview of Deeplay Internal Structure](https://github.com/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT181_internals.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/deeplay/blob/develop/tutorials/developers/DT181_internals.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

## Documentation

The detailed documentation of Deeplay is available at the following link: [https://deeptrackai.github.io/deeplay](https://deeptrackai.github.io/deeplay)

## Funding

This work was supported by the ERC Starting Grant ComplexSwimmers (Grant No. 677511), the ERC Starting Grant MAPEI (101001267), and the Knut and Alice Wallenberg Foundation.
