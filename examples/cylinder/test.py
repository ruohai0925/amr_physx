import torch
import torch.nn as nn



# x = torch.randn(1, 4, 16, 16)  # batch of 1, 4-channel images, each of size 16x16 pixels

# conv_layer = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=2, padding=1, padding_mode='replicate')
# output = conv_layer(x)

# print(output.shape)  # prints torch.Size([1, 16, 8, 8])



# Create a random input tensor with 128 channels and size 16x16 pixels
x = torch.randn(1, 128, 16, 16)

# Define the upsampling layer
upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

# Define the convolutional layer
conv_layer = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, padding_mode='replicate')

# Pass the input tensor through the upsampling and convolutional layers
upsampled = upsample_layer(x)
output = conv_layer(upsampled)

# Check the shape of the output tensor
print(output.shape)  # prints torch.Size([1, 64, 32, 32])