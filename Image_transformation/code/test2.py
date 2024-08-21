import cv2
import numpy as np

## computer 

computer_og_img = cv2.imread('Template/PartA/original/computer.png',0)
affine_computer_img = cv2.imread('Template/PartA/affine/computer.png',0)
perspective_computer_img = cv2.imread('Template/PartA/perspective/computer.png',cv2.IMREAD_GRAYSCALE)
computer_og_img_rows, computer_og_img_cols = computer_og_img.shape[:2]


all_pts_computer_affine = np.loadtxt('Template/PartA/correspondances/affine/computer.csv', delimiter=',', skiprows=1).astype(np.float32)
# print(pts_computer_affine)
pts_computer_affine_original = np.array(all_pts_computer_affine[0:3, 0:2]) # affine needs 3 points
pts_computer_after_affine = np.array(all_pts_computer_affine[0:3, 2:4]) # affine needs 3 points
# print(pts_computer_affine_original)
# print(pts_computer_after_affine)
affine_mat_computer= cv2.getAffineTransform(pts_computer_affine_original,pts_computer_after_affine)
# print(affine_mat_computer)
affine_on_computer = cv2.warpAffine(computer_og_img, affine_mat_computer, (computer_og_img_cols, computer_og_img_rows)) ## dont know if this line is needed.
# print(affine_on_computer)
final_comp = affine_computer_img - affine_on_computer
print(final_comp)


# ----------------
# You need at least 3 pairs of matching points for an affine transformation
# matching_points_original = np.array([[x1, y1], [x2, y2], [x3, y3], ...], dtype=np.float32)
# matching_points_transformed = np.array([[x1_t, y1_t], [x2_t, y2_t], [x3_t, y3_t], ...], dtype=np.float32)

# Add a homogeneous coordinate (1) to each point
matching_points_original_homogeneous = np.hstack((pts_computer_affine_original, np.ones((pts_computer_affine_original.shape[0], 1), dtype=np.float32)))
matching_points_transformed_homogeneous = np.hstack((pts_computer_after_affine, np.ones((pts_computer_after_affine.shape[0], 1), dtype=np.float32)))
# print(matching_points_original_homogeneous)
# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(matching_points_original_homogeneous, full_matrices=True)

# Calculate the pseudo-inverse of S, replacing small singular values with zeros
S_pseudo_inverse = np.zeros_like(S)
threshold = 1e-6
S_pseudo_inverse[S > threshold] = 1 / S[S > threshold]

# Calculate the transformation matrix
transformation_matrix = np.dot(U, np.dot(np.diag(S_pseudo_inverse), Vt))
print("Transformation Matrix:")
print(transformation_matrix)

# pts_computer_pers = np.loadtxt('Template/PartA/correspondances/perspective/computer.csv', delimiter=',', skiprows=1).astype(np.float32)
# # print(pts_computer_pers)
# pts_computer_pers_original = np.array(pts_computer_pers[0:4, 0:2]) # Persp needs 4 points
# pts_computer_after_persp = np.array(pts_computer_pers[0:4, 2:4]) # Persp needs 4 points
# # print(pts_computer_pers_original)
# # print(pts_computer_after_persp)
# perspective_mat_computer = cv2.getPerspectiveTransform(pts_computer_pers_original, pts_computer_after_persp)
# # print(perspective_mat_computer)
# perspective_on_computer = cv2.warpPerspective(computer_og_img, perspective_mat_computer, (computer_og_img_cols, computer_og_img_rows)) ## dont know if this line is needed.
# # print(perspective_on_computer)


