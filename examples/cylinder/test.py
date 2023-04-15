import torch
import torch.nn as nn



# x = torch.randn(1, 4, 16, 16)  # batch of 1, 4-channel images, each of size 16x16 pixels

# conv_layer = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate')
# output = conv_layer(x)

# print(output.shape)  # prints torch.Size([1, 16, 8, 8])



# # Create a random input tensor with 128 channels and size 16x16 pixels
# x = torch.randn(1, 128, 16, 16)

# # Define the upsampling layer
# upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

# # Define the convolutional layer
# conv_layer = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate')

# # Pass the input tensor through the upsampling and convolutional layers
# upsampled = upsample_layer(x)
# output = conv_layer(upsampled)

# # Check the shape of the output tensor
# print(output.shape)  # prints torch.Size([1, 64, 32, 32])




# Create a random input tensor with shape (6, 4, 3, 64, 128)
states = torch.randn(6, 4, 3, 64, 128)

# Select the input for the first time step of each sequence
xin0 = states[:, 0].to('cuda')  # move the resulting tensor to the GPU

# Check the shape and device of the resulting tensor
print(xin0.shape)  # prints torch.Size([6, 3, 64, 128])
print(xin0.device)  # prints cuda:0 (assuming the first GPU is used)


# Create a random tensor with shape (6, 128, 128)
kMatrix = torch.randn(6, 128, 128)

# Compute the sum of squared elements
sum_squared = torch.sum(torch.pow(kMatrix, 2))

# Check the result
print(sum_squared)  # prints a scalar value


