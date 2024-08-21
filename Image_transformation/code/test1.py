import cv2
import numpy as np

## computer 

computer_og_img = cv2.imread('Template/PartA/original/computer.png',0)
affine_computer_img = cv2.imread('Template/PartA/affine/computer.png',0)
perspective_computer_img = cv2.imread('Template/PartA/perspective/computer.png',0)
computer_og_img_rows, computer_og_img_cols = computer_og_img.shape[:2]


pts_computer_affine = np.loadtxt('Template/PartA/correspondances/affine/computer.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_computer_affine)
pts_computer_affine_original = np.array(pts_computer_affine[0:3, 0:2]) # affine needs 3 points
pts_computer_after_affine = np.array(pts_computer_affine[0:3, 2:4]) # affine needs 3 points
# print(pts_computer_affine_original)
# print(pts_computer_after_affine)
affine_mat_computer= cv2.getAffineTransform(pts_computer_affine_original,pts_computer_after_affine)
# print(affine_mat_computer)
affine_on_computer = cv2.warpAffine(computer_og_img, affine_mat_computer, (computer_og_img_cols, computer_og_img_rows)) ## dont know if this line is needed.
# print(affine_on_computer)

pts_computer_pers = np.loadtxt('Template/PartA/correspondances/perspective/computer.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_computer_pers)
pts_computer_pers_original = np.array(pts_computer_pers[0:4, 0:2]) # Persp needs 4 points
pts_computer_after_persp = np.array(pts_computer_pers[0:4, 2:4]) # Persp needs 4 points
# print(pts_computer_pers_original)
# print(pts_computer_after_persp)
perspective_mat_computer = cv2.getPerspectiveTransform(pts_computer_pers_original, pts_computer_after_persp)
# print(perspective_mat_computer)
perspective_on_computer = cv2.warpPerspective(computer_og_img, perspective_mat_computer, (computer_og_img_cols, computer_og_img_rows)) ## dont know if this line is needed.
# print(perspective_on_computer)



## Lena

lena_og_img = cv2.imread('Template/PartA/original/lena.png',0)
affine_lena_img = cv2.imread('Template/PartA/affine/lena.png',0)
perspective_lena_img = cv2.imread('Template/PartA/perspective/lena.png',0)
lena_og_img_rows, lena_og_img_cols = lena_og_img.shape[:2]


pts_lena_affine = np.loadtxt('Template/PartA/correspondances/affine/lena.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_lena_affine)
pts_lena_affine_original = np.array(pts_lena_affine[0:3, 0:2]) # affine needs 3 points
pts_lena_after_affine = np.array(pts_lena_affine[0:3, 2:4]) # affine needs 3 points
# print(pts_lena_affine_original)
# print(pts_lena_after_affine)
affine_mat_lena= cv2.getAffineTransform(pts_lena_affine_original,pts_lena_after_affine)
# print(affine_mat_lena)
affine_on_lena = cv2.warpAffine(lena_og_img, affine_mat_lena, (lena_og_img_cols, lena_og_img_rows)) ## dont know if this line is needed.
# print(affine_on_lena)

pts_lena_pers = np.loadtxt('Template/PartA/correspondances/perspective/lena.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_lena_pers)
pts_lena_pers_original = np.array(pts_lena_pers[0:4, 0:2]) # Persp needs 4 points
pts_lena_after_persp = np.array(pts_lena_pers[0:4, 2:4]) # Persp needs 4 points
# print(pts_lena_pers_original)
# print(pts_lena_after_persp)
perspective_mat_lena = cv2.getPerspectiveTransform(pts_lena_pers_original, pts_lena_after_persp)
# print(perspective_mat_lena)
perspective_on_lena = cv2.warpPerspective(lena_og_img, perspective_mat_lena, (lena_og_img_cols, lena_og_img_rows)) ## dont know if this line is needed.
# print(perspective_on_lena)

## Mario

mario_og_img = cv2.imread('Template/PartA/original/mario.jpg',0)
affine_mario_img = cv2.imread('Template/PartA/affine/mario.jpg',0)
perspective_mario_img = cv2.imread('Template/PartA/perspective/mario.jpg',0)
mario_og_img_rows, mario_og_img_cols = mario_og_img.shape[:2]


pts_mario_affine = np.loadtxt('Template/PartA/correspondances/affine/mario.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_mario_affine)
pts_mario_affine_original = np.array(pts_mario_affine[0:3, 0:2]) # affine needs 3 points
pts_mario_after_affine = np.array(pts_mario_affine[0:3, 2:4]) # affine needs 3 points
# print(pts_mario_affine_original)
# print(pts_mario_after_affine)
affine_mat_mario= cv2.getAffineTransform(pts_mario_affine_original,pts_mario_after_affine)
# print(affine_mat_mario)
affine_on_mario = cv2.warpAffine(mario_og_img, affine_mat_mario, (mario_og_img_cols, mario_og_img_rows)) ## dont know if this line is needed.
# print(affine_on_mario)

pts_mario_pers = np.loadtxt('Template/PartA/correspondances/perspective/mario.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_mario_pers)
pts_mario_pers_original = np.array(pts_mario_pers[0:4, 0:2]) # Persp needs 4 points
pts_mario_after_persp = np.array(pts_mario_pers[0:4, 2:4]) # Persp needs 4 points
# print(pts_mario_pers_original)
# print(pts_mario_after_persp)
perspective_mat_mario = cv2.getPerspectiveTransform(pts_mario_pers_original, pts_mario_after_persp)
# print(perspective_mat_mario)
perspective_on_mario = cv2.warpPerspective(mario_og_img, perspective_mat_mario, (mario_og_img_cols, mario_og_img_rows)) ## dont know if this line is needed.
# print(perspective_on_mario)



## Mountain

mountain_og_img = cv2.imread('Template/PartA/original/mountain.jpg',0)
affine_mountain_img = cv2.imread('Template/PartA/affine/mountain.jpg',0)
perspective_mountain_img = cv2.imread('Template/PartA/perspective/mountain.jpg',0)
mountain_og_img_rows, mountain_og_img_cols = mountain_og_img.shape[:2]


pts_mountain_affine = np.loadtxt('Template/PartA/correspondances/affine/mountain.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_mountain_affine)
pts_mountain_affine_original = np.array(pts_mountain_affine[0:3, 0:2]) # affine needs 3 points
pts_mountain_after_affine = np.array(pts_mountain_affine[0:3, 2:4]) # affine needs 3 points
# print(pts_mountain_affine_original)
# print(pts_mountain_after_affine)
affine_mat_mountain= cv2.getAffineTransform(pts_mountain_affine_original,pts_mountain_after_affine)
# print(affine_mat_mountain)
affine_on_mountain = cv2.warpAffine(mountain_og_img, affine_mat_mountain, (mountain_og_img_cols, mountain_og_img_rows)) ## dont know if this line is needed.
# print(affine_on_mountain)

pts_mountain_pers = np.loadtxt('Template/PartA/correspondances/perspective/mountain.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_mountain_pers)
pts_mountain_pers_original = np.array(pts_mountain_pers[0:4, 0:2]) # Persp needs 4 points
pts_mountain_after_persp = np.array(pts_mountain_pers[0:4, 2:4]) # Persp needs 4 points
# print(pts_mountain_pers_original)
# print(pts_mountain_after_persp)
perspective_mat_mountain = cv2.getPerspectiveTransform(pts_mountain_pers_original, pts_mountain_after_persp)
# print(perspective_mat_mountain)
perspective_on_mountain = cv2.warpPerspective(mountain_og_img, perspective_mat_mountain, (mountain_og_img_cols, mountain_og_img_rows)) ## dont know if this line is needed.
# print(perspective_on_mountain)



## water

water_og_img = cv2.imread('Template/PartA/original/water.jpg',0)
affine_water_img = cv2.imread('Template/PartA/affine/water.jpg',0)
perspective_water_img = cv2.imread('Template/PartA/perspective/water.jpg',0)
water_og_img_rows, water_og_img_cols = water_og_img.shape[:2]


pts_water_affine = np.loadtxt('Template/PartA/correspondances/affine/water.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_water_affine)
pts_water_affine_original = np.array(pts_water_affine[0:3, 0:2]) # affine needs 3 points
pts_water_after_affine = np.array(pts_water_affine[0:3, 2:4]) # affine needs 3 points
# print(pts_water_affine_original)
# print(pts_water_after_affine)
affine_mat_water= cv2.getAffineTransform(pts_water_affine_original,pts_water_after_affine)
# print(affine_mat_water)
affine_on_water = cv2.warpAffine(water_og_img, affine_mat_water, (water_og_img_cols, water_og_img_rows)) ## dont know if this line is needed.
# print(affine_on_water)

pts_water_pers = np.loadtxt('Template/PartA/correspondances/perspective/water.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_water_pers)
pts_water_pers_original = np.array(pts_water_pers[0:4, 0:2]) # Persp needs 4 points
pts_water_after_persp = np.array(pts_water_pers[0:4, 2:4]) # Persp needs 4 points
# print(pts_water_pers_original)
# print(pts_water_after_persp)
perspective_mat_water = cv2.getPerspectiveTransform(pts_water_pers_original, pts_water_after_persp)
# print(perspective_mat_water)
perspective_on_water = cv2.warpPerspective(water_og_img, perspective_mat_water, (water_og_img_cols, water_og_img_rows)) ## dont know if this line is needed.
# print(perspective_on_water)





