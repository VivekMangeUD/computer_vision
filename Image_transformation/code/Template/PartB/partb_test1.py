import cv2
import numpy as np

# def custom_filter2D(image, kernel):
#     image_height, image_width = image.shape
#     kernel_height, kernel_width = kernel.shape
#     padding_y = kernel_height // 2
#     padding_x = kernel_width // 2
#     output_image = np.zeros((image_height, image_width), dtype=np.uint8) # empty image

#     # Iterate over each pixel in the input image
#     for y in range(padding_y, image_height - padding_y):
#         for x in range(padding_x, image_width - padding_x):
#             roi = image[y - padding_y:y + padding_y + 1, x - padding_x:x + padding_x + 1]
#             weighted_sum = np.sum(roi * kernel)  # add value
#             output_image[y, x] = weighted_sum  # store at centre 

#     return output_image

# image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)  # grayscale

# kernel = np.array([
#   [1, 1, 1],
#   [1, 1, 1],
#   [1, 1, 1]
# ]) / 9

# convolved_image_custom = custom_filter2D(image, kernel)
# # convolved_image = cv2.filter2D(image, -1, kernel)
# cv2.imshow('Convolved Image (Custom)', convolved_image_custom)
# # cv2.imshow('Convolved Image ', convolved_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # separable 1D Gaussian kernels

###########################################################################


def convolution_rgb(image, kernel):
    height, width, _ = image.shape
    k_height, k_width = kernel.shape
    output = np.zeros((height, width, 3), dtype=np.uint8)

    pad_height = k_height // 2
    pad_width = k_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')

    # Perform convolution
    for y in range(height):
        for x in range(width):
            roi = padded_image[y:y + k_height, x:x + k_width, :]
            for channel in range(3):
                output[y, x, channel] = np.sum(roi[:, :, channel] * kernel)
    output = np.clip(output, 0, 255)
    return output

if __name__ == "__main__":
    # image = np.array([
    #     [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    #     [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
    #     [[128, 128, 128], [0, 0, 0], [255, 255, 255]]
    # ], dtype=np.uint8)
    image = cv2.imread('lena.png')  # grayscale

    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # Perform convolution
    convolved_image = convolution_rgb(image, kernel)
    convolved_image_cv = cv2.filter2D(image, -1, kernel)
    cv2.imshow('Convolved Image (Custom)', convolved_image)
    cv2.imshow('Convolved Image ', convolved_image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Print the result
    # print(convolved_image)
