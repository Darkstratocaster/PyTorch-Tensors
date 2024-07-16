import torch

x = torch.empty(2, 2, 3)
y = torch.ones(2,3, dtype=torch.float16)
z = torch.rand(6, 9)
q = torch.zeros(3,7)
d = torch.tensor([2.5, 0.6])



print("Tensor x with uninitialized data: ")
print(x)
print("\nTensor y filled with scalar value 1: ")
print(y)
print("\nTensor z with random values with interval 6,9: ")
print(z)
print("\nTensor q filled with scalar value 0: ")
print(q)

print("\nTensor y:s datatype: ")
print(y.dtype)
print("\nTensor y:s size:")
print(y.size())
print("\nTensor d with user-defined values: ")
print(d)

a = torch.rand(2,2)
b = torch.rand(2,2)
c = a + b
c = torch.add(a,b)

print("\nSum of two tensors with random values with interval 2,2 is:")
print(c)

print("\nSame can also be done with the function add_")
b.add_(a)
print(b)


print("\nSubstraction of the aforementioned tensors:")
e = a - b
e = torch.sub(a, b)
print(e)

print("\nMultiplying the aforementioned tensors:")
t = a.mul_(b)
print(t)

print("\nDividing the aforementioned tensors:")
o = a / b
o = torch.div(a, b)

sliced = torch.rand(2, 3)
print("\n Randomized tensor:")
print(sliced)
print("\n Randomized tensors first elements of every row (the tensor is sliced):")
print(sliced[:, 0])
print("\n Randomized tensors item at location 1,1:")
print(sliced[1, 1].item())

print("\n Four-column trensor displayed as a one-column tensor:")
four_columns = torch.rand(4,4)
one_column = four_columns.view(16)
print(one_column)

print("\n Four-column trensor displayed as a two-column tensor:")
two_columns = four_columns.view(2, 8)
print(two_columns)

