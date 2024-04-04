This section is not complete yet. We are working on it. Current applications do _not_ follow these guidelines yet.

Applications can be hard to test because they often require a lot of data and time to train to fully validate. This is not feasible, since we want to retain the unit test speed to within a few minutes. Here are some guidelines to follow when testing applications:

- Test both using build and create methods.
  - Validate that the forward pass works as expected.
  - Validate that the loss and metrics are correctly set.
  - Validate that the optimizer is correctly set.
    - This can be done by manually calling `app.configure_optimizer()` and checking that returned optimizer is correct the expected one.
    - Also, verify that the parameters of the optimizer are correctly set. This can
      be done by doing a forward pass, backward pass, and then checking that the optimizer's parameters have been updated.
      - If there are multiple optimizers with different parameters, ensure that the correct parameters are updated.
- Test `app.compute_loss` on a small input and verify that the loss is correct.
- Test `app.training_step` on a small input and verify that the loss is correct. Note that you might need to attach a trainer for. This can simply be done by creating a mock
  trainer and setting `app.trainer = mock_trainer`. We will provide a more detailed example in the future.
- Test training the application on a single epoch both using `app.fit` and `trainer.fit(app)`.
  - It's a good idea to check that the training history contains the correct keys.
  - Use as small a dataset and model as possible to keep the test fast. Turn off checkpointing and logging to speed up the test.
- Test that the application can be saved and loaded correctly.
  - Currently, we only guarantee `save_state_dict` and `load_state_dict` to work correctly. A good way to test is to save the state dict, create a new application, load the state dict, and then check that the new application has the same state as the old one.
- Test any application-specific methods, such as `app.detect`, `app.classify`, etc.
