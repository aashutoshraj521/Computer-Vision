import numpy as np
import torch

array = np.array([
    [1,2],
    [3,4]
])

tensor = torch.from_numpy(array)

print(tensor)

array[0,0] = 100
print(array)
print(tensor)

tensor = torch.tensor([
    [5,6],
    [7,8]
])
print(tensor.device)
tensor = tensor.cpu()
array = tensor.numpy()
print(array)
print(tensor.dtpye)
print(array.dtype)

import cv2
image = cv2.imread("G:\My Drive\Colab Notebooks\Computer Vision\Images\mahadev.jpg")
print(type(image))
print(image.shape)

tensor = torch.from_numpy(image)
tensor = tensor.permute(2,0,1)
print(type(tensor))
print(tensor.shape)