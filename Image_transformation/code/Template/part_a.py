import os
import cv2 as cv
import numpy as np


'''
This is a sample source code file.
It is not complete and does not work.
It just gives you an overall idea.
'''

for file_name in files: # Loop through images
    original_image = cv.imread(file_name)

     # Read the points
    original_points =
    affine_points =
    perspective_points =

    # Use the built-in OpenCV functions
    affine_transform_builtin = cv.getAffineTransform()
    perspective_transform_builtin = cv.warpAffine()

    # Use your own implemented functions
    affine_transform_method_a = get_affine_transform_method_a(original_points, affine_points)
    affine_image_method_a = warp_affine(original_image, affine_transform_method_a)

    affine_transform_method_b = get_affine_transform_method_b(original_points, affine_points)
    affine_image_method_b = warp_affine(original_image, affine_transform_method_b)

    perspective_transform_method_a = get_perspective_transform_method_a(original_points, perspective_points)
    perspective_image_method_a = warp_perspective(original_image, perspective_transform_method_a)

    perspective_transform_method_b = get_perspective_transform_method_b(original_points, perspective_points)
    perspective_image_method_b = warp_perspective(original_image, perspective_transform_method_b)

    affine_error_method_a = calculate_error(affine_transform_method_a, affine_transform_builtin)
    affine_error_method_b = calculate_error(affine_transform_method_b, affine_transform_builtin)
    perspective_error_method_a = calculate_error(perspective_transform_method_a, perspective_transform_builtin)
    perspective_error_method_b = calculate_error(perspective_transform_method_b, perspective_transform_builtin)


    print('############################################################')
    print('Processed image:', file_name)
    print('------------------------------------------------------------')
    print('Affine transformation matrix, method A\n', affine_transform_method_a)
    print('------------------------------------------------------------')
    print('Affine transformation matrix, method B\n', affine_transform_method_b)
    print('------------------------------------------------------------')
    print('Perspective transformation matrix, method A\n', perspective_transform_method_a)
    print('------------------------------------------------------------')
    print('Perspective transformation matrix, method B\n', perspective_transform_method_b)
    print('------------------------------------------------------------')
    print('Error between my affine vs. built-in function, method A\n', affine_error_method_a)
    print('------------------------------------------------------------')
    print('Error between my affine vs. built-in function, method B\n', affine_error_method_b)
    print('------------------------------------------------------------')
    print('Error between my perspective vs. built-in function, method A\n', perspective_error_method_a)
    print('------------------------------------------------------------')
    print('Error between my perspective vs. built-in function, method B\n', perspective_error_method_b)
    print('------------------------------------------------------------')


    # save generated images to ./outputs folder
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    os.chdir('outputs')

    cv.imwrite(file_name.rsplit('.', maxsplit=1)[0] + '_affine_method_a.png', affine_image_method_a)
    cv.imwrite(file_name.rsplit('.', maxsplit=1)[0] + '_affine_method_b.png', affine_image_method_b)
    cv.imwrite(file_name.rsplit('.', maxsplit=1)[0] + '_perspective_method_a.png', perspective_transform_method_a)
    cv.imwrite(file_name.rsplit('.', maxsplit=1)[0] + '_perspective_method_b.png', perspective_transform_method_b)
