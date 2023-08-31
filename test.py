# #%%
# import deeptorch as D

# config = D.Config().socket.inputs[0](1)

# # %%
# config._rules[0]
# # %%

# class C:
#     ...

# class A:
#     ...

# class B(C):
#     def __init__(self, x):
#         self.x = x

#     def func(self):
#         return self.x

# # %%
# a = A()
# b = B(1)
# # %%
# a.__dict__ = b.__dict__
# a.__class__ = b.__class__
# # %%
# a.func()
# # %%
# class C:
#     def method_c(self):
#         return "This is method C."
# class A:
#     def method_a(self):
#         return "This is method A."
#     def __repr__(self):
#         return "This is A."
# class B(C):
#     def method_b(self):
#         return "This is method B."
#     def __repr__(self):
#         return "This is B."
    
# a_instance = A()
# b_instance = B()

# # Copying __dict__ and changing __class__
# a_instance.__dict__ = b_instance.__dict__
# a_instance.__class__ = B

# # print(a_instance.method_a())  # This works
# print(a_instance.method_b())  # This raises an AttributeError
# print(a_instance.method_c())  # This raises an AttributeError
# # %%

# a_instance.__repr__()
# # %%
