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
