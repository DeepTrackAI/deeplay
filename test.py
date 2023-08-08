# # %%

# class Test:

#     def __new__(cls, *args, config=None, **kwargs):
#         print("new")
#         obj = super().__new__(cls)
#         obj.__init__(*args, **kwargs)
#         obj.__init__ = lambda *args, **kwargs: None
#         return obj
    
#     def __init__(self):
#         print("init")


# Test(config=1)
# # %%
