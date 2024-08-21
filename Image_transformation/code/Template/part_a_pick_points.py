import cv2 as cv
import numpy as np

# get mouse pixel postions

original_image = cv.imread('PartA\\original\\lena.png')
affine_image = cv.imread('PartA\\affine\\lena.png')
perspective_image = cv.imread('PartA\\perspective\\lena.png')


def original(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'[Original] ({x}, {y})')


def affine(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'[Affine] ({x}, {y})')


def perspective(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'[Perspective] ({x}, {y})')


cv.namedWindow('Original Image')
cv.namedWindow('Affine Image')
cv.namedWindow('Perspective Image')
cv.setMouseCallback('Original Image', original)
cv.setMouseCallback('Affine Image', affine)
cv.setMouseCallback('Perspective Image', perspective)
cv.imshow('Original Image', original_image)
cv.imshow('Affine Image', affine_image)
cv.imshow('Perspective Image', perspective_image)
k = cv.waitKey()
cv.destroyAllWindows()
