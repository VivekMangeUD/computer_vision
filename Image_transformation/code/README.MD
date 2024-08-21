# Transformation Matrix Estimation Project

**Author:** Vivek Mange

This project involves estimating affine and perspective transformation matrices using corresponding points between original and transformed images. The goal is to manually implement matrix estimation methods and compare them with built-in functions.

### Tasks and Concepts:

**Task A: Solve with Minimal Correspondences**

1. **Affine Transformation Matrix (X1):**
   - **Concept:** An affine transformation includes scale, rotation, and shear. It can be estimated using at least 3 corresponding points between the original and transformed images.
   - **Steps:**
     1. Use the given corresponding points to set up a linear system.
     2. Solve for the affine transformation matrix using the Least Squares method or Singular Value Decomposition (SVD).
     3. Compare the estimated matrix with the result from OpenCV’s `getAffineTransform()` function and compute the error.

2. **Perspective Transformation Matrix (X2):**
   - **Concept:** A perspective transformation maps points from a 3D plane to a 2D plane and requires at least 4 corresponding points.
   - **Steps:**
     1. Set up a linear system using the 4 corresponding points.
     2. Solve for the perspective transformation matrix using Least Squares or SVD.
     3. Compare the estimated matrix with the result from OpenCV’s `getPerspectiveTransform()` function and compute the error.

**Task B: Solve with Additional Correspondences**

1. **Affine Transformation Matrix (X1):**
   - **Concept:** Using more than 3 points (over-constrained system) can improve the accuracy of the affine transformation estimation.
   - **Steps:**
     1. Use more than 3 corresponding points to set up an over-constrained linear system.
     2. Estimate the affine matrix with this additional data.
     3. Compare the result with the built-in function and compute the error.

2. **Perspective Transformation Matrix (X2):**
   - **Concept:** Using more than 4 points provides a better estimation for the perspective transformation matrix.
   - **Steps:**
     1. Use more than 4 corresponding points to set up an over-constrained linear system.
     2. Estimate the perspective matrix with this additional data.
     3. Compare the result with the built-in function and compute the error.

**Extra Correspondences (for `computer.png`):**

1. **Manually Pick 10 Additional Correspondences:**
   - **Concept:** Adding extra correspondences helps in refining the transformation estimates.
   - **Steps:**
     1. Manually select 10 corresponding points between the original and transformed images.
     2. Save these points in a new .csv file.

2. **Re-estimate Matrices:**
   - **Concept:** Recalculate X1 and X2 using the additional points.
   - **Steps:**
     1. Use the new correspondences to estimate X1 and X2.
     2. Compute and report errors and transformed images using these updated matrices.

### Data:

- **Original Images:** Located in `PartA/original`.
- **Transformed Images:** Found in `PartA/affine` and `PartA/perspective`.
- **Correspondences:** Available in `PartA/correspondences`.

### Deliverables:

For each original image:

1. **Affine Transformation (X1):**
   - **Method A:** Estimate using 3 correspondences.
   - **Method B:** Estimate using more than 3 correspondences.
   - **Deliverables:** Transformed images and errors compared to the built-in function.

2. **Perspective Transformation (X2):**
   - **Method A:** Estimate using 4 correspondences.
   - **Method B:** Estimate using more than 4 correspondences.
   - **Deliverables:** Transformed images and errors compared to the built-in function.

3. **Extra Correspondences (computer.png):**
   - **Manually Picked Points:** Report the estimated matrices and errors with the additional correspondences.

### Implementation:

- Implement matrix estimation using Least Squares or SVD.
- Compare your results with OpenCV’s built-in functions.
- Transform and display images using your estimated matrices.

