#%%
import deeplay as dl
import torch

CNN_template = dl.Sequential(
    dl.Layer(torch.nn.Conv2d, in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1),
    dl.Layer(torch.nn.AdaptiveAvgPool2d,output_size = 1),
    dl.Layer(torch.nn.Flatten),
)
# CNN_template[0].configure(torch.nn.MaxPool2d, kernel_size = 2)
#%%
CNN = CNN_template.new()
CNN
# CNN_classifier_template = dl.BinaryClassifier(
#     model=CNN,
#     optimizer=dl.Adam(lr=.001),
# )
# CNN_classifier_template.build()
# print(CNN, CNN._actual_init_args)
# %%
self = CNN_template
user_config = self._collect_user_configuration()
args = self._actual_init_args["args"]
_args = self._args
kwargs = self._actual_init_args["kwargs"]

obj = type(self).__new__(type(self), *args, _args=_args, **kwargs)
obj.__pre_init__(obj, *args, _args=_args, **kwargs)
obj._args,  _args

# obj._args
# %%
