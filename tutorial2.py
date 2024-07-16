import torch
import numpy as np

a = torch.ones(5)
print("A five-element tensor with ones:")
print(a)
b = a.numpy()
print("\n Aforementioned tensor turned into a Numpy ndarray:")
print(type(b))

a.add_(1)
print("\nThe Numpy array points to the same memory location as the first tensor - if you change the values of a, b changes too:")
print(a)
print(b)