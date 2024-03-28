# Deeplay Applications

Applications are broadly defined as classes that represent a task such as classification, without depending heavily on the exact architecture. They are the highest level of abstraction in the Deeplay library. Applications are designed to be easy to use and require minimal configuration to get started. They are also designed to be easily extensible, so that you can add new features without having to modify the existing code.

## What's in an application?

As a general rule of thumb, try to minimize the number of models in an application. Best is if there is a single model, accessed as `app.model`. Some applications require more,
such as `gan.generator` and `gan.discriminator`. This is fine, but try to keep it to a minimum. A bad example would be for a classifier to include `app.conv_backbone`, `app.conv_to_fc_connector` and `app.fc_head`. This is bad because it limits the flexibility of the application to architectures that fit this exact structure. Instead, the application should have a single model that can be easily replaced with a different model.

### Training

The primary function of an application is to define how it is trained. This includes the loss function, the optimizer, and the metrics that are used to evaluate the model. Applications also define how the model is trained, including the training loop, the validation loop, and the testing loop. Applications are designed to be easy to use, so that you can get started quickly without having to worry about the details of the training process.

The training step is, broadly, defined as follows:

```python
x, y = self.train_preprocess(batch)
y_hat = self(x)
loss = self.compute_loss(y_hat, y)
# logging
```

If the training can be defined in this way, then you can implement the `train_preprocess`, `compute_loss`, and `forward` methods to define the training process. If the training process is more complex, then you can override the `training_step` method to define the training process. The default behavior of `train_preprocess` is the identity function, and the default behavior of `compute_loss` is to call `self.loss(y_hat, y)`.

`train_preprocess` is intended to apply any operations that cannot be simply defined as a part of the dataset. For example, in some `self-supervised` models, the target is calculated from the input data. This can be done here. It can also be used to ensure the dtype of the input matches the expected dtype of the model. Most likely, you will not need to override this method.

`compute_loss` is intended to calculate the loss of the model. This can be as simple as calling `self.loss(y_hat, y)`, or it can be more complex. It is more likely that you will need to override this method.

If you need to define a custom training loop, you can override the `training_step` method entirely. This method is called for each batch of data during training. It should return the loss for the batch. Note that if you override the `training_step` method, you will to handle the logging of the loss yourself. This is done by calling `self.log('train_loss', loss, ...)` where `...` is any setting you want to pass to the logger (see `lightning.LightningModule.log` for more information).
