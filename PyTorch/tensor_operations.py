import torch
A = torch.tensor([1, 2, 3, 4])
B = torch.tensor([5, 6, 7, 8])
print("\nThe sum is: ",A+B)
print("The subtraction is: ", A-B)
print("The element wise multiplication is: ", A*B)

# Shape is not compatible but still matrix multiplication
print("\nThe matrix multiplication is: ", torch.matmul(A, B))

A = A.reshape(2,2)
print("A: ", A)
B = B.reshape(2,2)
print("B: ", B)
print("The matrix multiplication is:\n", torch.matmul(A, B))

A = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print("\nA[0] first row:\n", A[0])
print("\nA[1,2] single element: ", A[1,2])
print("\nfirst 2 rows A[:2]:\n", A[:2])
print("\nsecond column A[:,1]:\n", A[:,1])

A += 10
print("\nA + 10:\n", A)

A = A.reshape(-1)
print("\nAfter flattening: A\n", A)