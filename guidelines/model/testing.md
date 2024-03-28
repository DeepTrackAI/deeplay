This section is not complete yet. We are working on it. Current models do _not_ follow these guidelines yet.

Models are easier to test than applications because they are usually smaller and have fewer dependencies. We do not test if models can be trained.

- Test that the model, created with default arguments, can be created.
- Test that the model, created with default arguments, has the correct number of parameters.
- Test that the model, created with default arguments, can be saved and loaded using `save_state_dict` and `load_state_dict`.
- test that the model, created with default arguments, has the expected hierarchical structure. This is mostly for forward compatibility.
- Test the forward pass with a small input and verify that the output is correct.
- Test that the model can be created with non-default arguments and that the arguments are correctly set.
