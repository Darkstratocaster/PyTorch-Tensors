import torch
import numpy as np

a = np.ones(5)
print("Numpy-ndarray with ones:")
print(a)
b = torch.from_numpy(a)
print("\n Tensor made from the ndarray with the same memory address:")
print(b)

a += 1
print("\n Summing the numpy-array changes the tensor too, because of the memory address")
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y=torch.ones(5)
    y = y.to(device)
    z = x + y
    # z.numpy() would return an error, since numpy can only handle cpu tensors, take it to cpu like this
    z = z.to("cpu")
    print("\n Tensor taken to the gpu:")
    print(x)