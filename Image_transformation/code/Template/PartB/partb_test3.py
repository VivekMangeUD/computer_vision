# import numpy as np
# import cv2

# def convolve_image(input_image, kernel):
#     # Get dimensions of the input image and kernel
#     input_height, input_width, _ = input_image.shape
#     kernel_height, kernel_width = kernel.shape

#     # Calculate the padding required for "valid" convolution
#     pad_height = kernel_height // 2
#     pad_width = kernel_width // 2

#     # Create an output image with the same dimensions as the input
#     output_image = np.zeros_like(input_image)

#     # Pad the input image with zeros
#     padded_image = np.pad(input_image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

#     # Perform convolution
#     for i in range(input_height):
#         for j in range(input_width):
#             for k in range(3):  # Loop over RGB channels
#                 output_image[i, j, k] = np.sum(
#                     padded_image[i:i+kernel_height, j:j+kernel_width, k] * kernel
#                 )

#     # Ensure pixel values are within [0, 255] range
#     output_image = np.clip(output_image, 0, 255)

#     return output_image

# # Example usage:
# if __name__ == "__main__":
#     # Load an example image
#     input_image = cv2.imread('lena.png')

#     # Define a sample kernel (e.g., a 3x3 Gaussian blur kernel)
#     # kernel = np.array([[1, 2, 1],
#     #                    [2, 4, 2],
#     #                    [1, 2, 1]])

#     kernel = np.array([[-1, -1, -1],
#                     [-1,  8, -1],
#                     [-1, -1, -1]])

#     # Normalize the kernel
#     kernel = kernel / np.sum(kernel)

#     # Perform convolution
#     output_image = convolve_image(input_image, kernel)
#     cv_col = cv2.filter2D(input_image, -1, kernel)

#     # Display the output image
#     cv2.imshow('Output Image', output_image)
#     cv2.imshow('cv2 image ', cv_col)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# import numpy as np
# import cv2

# def convolve_image(input_image, kernel):
#     # Get dimensions of the input image and kernel
#     input_height, input_width, _ = input_image.shape
#     kernel_height, kernel_width = kernel.shape

#     # Calculate the padding required for "valid" convolution
#     pad_height = kernel_height // 2
#     pad_width = kernel_width // 2

#     # Create an output image with the same dimensions as the input
#     output_image = np.zeros_like(input_image)

#     # Pad the input image with zeros
#     padded_image = np.pad(input_image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

#     # Perform convolution
#     for i in range(input_height):
#         for j in range(input_width):
#             for k in range(3):  # Loop over RGB channels
#                 output_image[i, j, k] = np.sum(
#                     padded_image[i:i+kernel_height, j:j+kernel_width, k] * kernel
#                 )

#     # Normalize the output image to the [0, 255] range
#     output_image = np.clip(output_image, 0, 255)

#     return output_image

# # Example usage:
# if __name__ == "__main__":
#     # Load an example image
#     input_image = cv2.imread('lena.png')

#     # Define the Laplacian kernel
#     kernel = np.array([[-1, -1, -1],
#                        [-1,  8, -1],
#                        [-1, -1, -1]])
# #     # kernel = np.array([[1, 2, 1],
# #     #                    [2, 4, 2],
# #     #                    [1, 2, 1]])
#     # Normalize the kernel
#     kernel = kernel / np.sum(kernel)

#     # Perform convolution
#     output_image = convolve_image(input_image, kernel)
#     cv_col = cv2.filter2D(input_image, -1, kernel)

#     cv2.imshow('cv2 image ', cv_col)

#     # Display the output image
#     cv2.imshow('Output Image', output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

###############################################

import numpy as np
import cv2

image = cv2.imread('lena.png')
print('image shape ', image.shape)

#  Define the size of the matrix
width, height, channels = 7, 7, 3
array = np.arange(27).reshape(3, 3, 3)

print(array)
# Generate a random integer matrix of the specified size
# random_matrix = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
# print('random image shape ', random_matrix.shape)
# print(random_matrix[0].shape)
kernel = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]])


# Define the amount of padding for each side
top_padding = 1
bottom_padding = 1
left_padding = 1
right_padding = 1

# Add padding to the matrix
# padded_matrix = np.pad(matrix, ((top_padding, bottom_padding), (left_padding, right_padding)), mode='constant', constant_values=0)
# padded_matrix = np.pad(random_matrix, ((top_padding, bottom_padding), (left_padding, right_padding)), mode='constant', constant_values=0)
# print('test image shape ', padded_matrix.shape)

# print(padded_matrix)

# print(image[0].shape)

#########################

# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('lena.png')  # Replace 'your_image_path.jpg' with the actual image file path
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# # Define the 3x3 kernel
# kernel = np.array([[-1, -1, -1],
#                     [-1,  8, -1],
#                     [-1, -1, -1]])

# # Get the dimensions of the image and kernel
# image_height, image_width, _ = image.shape
# kernel_height, kernel_width = kernel.shape

# # Initialize the result (convolved) image as zeros
# result = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1, 3), dtype=np.uint8)

# # Perform convolution
# for i in range(image_height - kernel_height + 1):
#     for j in range(image_width - kernel_width + 1):
#         # Extract a region from the image
#         region = image[i:i + kernel_height, j:j + kernel_width, :]
#         # Perform element-wise multiplication and sum
#         conv_value = np.sum(region * kernel)

#         # Set the result value in the output image
#         result[i, j] = [conv_value, conv_value, conv_value]

# # Display the original and convolved images
# cv_convo = cv2.filter2D(image, -1, kernel)

# cv2.imshow('Original Image', image)
# cv2.imshow('Convolved Image', result)
# cv2.imshow('cv convo', cv_convo)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
