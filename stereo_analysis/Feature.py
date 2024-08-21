import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided     

def matching_score_def(image, template, process, mask=None):
    if mask is None:
        mask = np.ones_like(template)
    window_size = template.shape

    updatedImage = as_strided(image, shape=(image.shape[0] - window_size[0] + 1, image.shape[1] - window_size[1] + 1,) +
                                           window_size, strides=image.strides * 2)

    if process == 1:
        matching_score = ((abs(updatedImage - template)) * mask).sum(axis=-1).sum(axis=-1)
    elif process == 2:
        matching_score = ((updatedImage - template) ** 2 * mask).sum(axis=-1).sum(axis=-1)
    return matching_score

def disparity(left, right, templateSize, window, lambdaValue, process):
    im_rows, im_cols = left.shape
    tpl_rows = tpl_cols = templateSize
    disparity = np.zeros(left.shape, dtype=np.float32)


    for r in range(tpl_rows//2, im_rows-tpl_rows//2):
        tr_min, tr_max = max(r-tpl_rows//2, 0), min(r+tpl_rows//2+1, im_rows)
        for c in range(tpl_cols//2, im_cols-tpl_cols//2):
            tc_min = max(c-tpl_cols//2, 0)
            tc_max = min(c+tpl_cols//2+1, im_cols)
            tpl = left[tr_min:tr_max, tc_min:tc_max].astype(np.float32)
            rc_min = max(c - window // 2, 0)
            rc_max = min(c + window // 2 + 1, im_cols)
            R_strip = right[tr_min:tr_max, rc_min:rc_max].astype(np.float32)
            if process ==3 :
                error = cv2.matchTemplate(R_strip, tpl, method=cv2.TM_CCORR_NORMED)
            else:
                error = matching_score_def(R_strip, tpl, process)
            c_tf = max(c-rc_min-tpl_cols//2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error + np.abs(dist * lambdaValue)
            _,_,min_loc,_ = cv2.minMaxLoc(cost)
            disparity[r, c] = dist[min_loc[0]]
    return disparity


def findCorners(I, window_size, k, thresh):
    dy, dx = np.gradient(I)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    h, w = I.shape
    corners = []
    copiedImage = I.copy()
    final_img = cv2.cvtColor(copiedImage, cv2.COLOR_GRAY2RGB)
    offset = window_size//2

    #looping through the images and detecting the corners
    for y in range(offset, h-offset):
        for x in range(offset, w-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #computing corner response using determinant and trace
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            #if corner response crosses threshold, point is marked
            if r > thresh:
                #print x, y, r
                corners.append([x, y, r])
                final_img.itemset((y, x, 0), 0)
                final_img.itemset((y, x, 1), 0)
                final_img.itemset((y, x, 2), 255)
    return final_img, corners

def resolution(image, levels):
    h, w, c = image.shape
    outputImage = image
    for i in range(0, h, 2**(levels-1)):
        for j in range(0, w, 2**(levels-1)):
            for k in range(0, 3, 1):
                outputImage[i:i+2**(levels-1), j:j+2**(levels-1), k] = image[i, j, k]
    return outputImage


def main():
    levels = int(input('Levels for multi-resolution : '))
    templateSize = int(input('Template size : '))
    window = int(input('Window size : '))
    process = int(input('SAD: 1 SSD: 2 NCC: 3 \n choose : '))

    leftImage = str('5_left.ppm') # Image name change for different images format 5_left.ppm
    rightImage = str('5_right.ppm') # Image name change for different images format 5_right.ppm

    original_left_img = cv2.imread(leftImage) 
    original_right_img = cv2.imread(rightImage) 

    cv2.imshow('1st ', original_left_img)
    cv2.imshow('2nd ', original_right_img)
    if original_left_img is None:
        print('enpty image')
    left_resol_img = resolution(original_left_img, levels)
    right_resol_img = resolution(original_right_img, levels)

    # cv2.imwrite('output/Feature/5/left resol.jpg', left_resol_img)
    # cv2.imwrite('output/Feature/5/right resol.jpg', right_resol_img)

    gray_left = cv2.cvtColor(left_resol_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_resol_img, cv2.COLOR_BGR2GRAY)

    window_size = 3
    k = 0.15
    thresh = 100000
    left_corner, corners = findCorners(gray_left, int(window_size), float(k), int(thresh))
    right_corner, corners = findCorners(gray_right, int(window_size), float(k), int(thresh))

    # cv2.imwrite('output/Feature/5/left featured image.jpg', left_corner)
    # cv2.imwrite('output/Feature/5/right featured image.jpg', right_corner)
    cv2.imshow('Corner Response Left Image SAD', left_corner)
    cv2.imshow('Corner Response Right Image SAD', right_corner)
    finalleft = cv2.cvtColor(left_corner, cv2.COLOR_RGB2GRAY)
    finalright = cv2.cvtColor(right_corner, cv2.COLOR_RGB2GRAY)


    if process == 1:
        leftDisparity = np.abs(disparity(finalleft, gray_right, templateSize=templateSize, window=window, lambdaValue=0.0, process=1))
        rightDisparity = np.abs(disparity(finalright, gray_left, templateSize=templateSize, window=window, lambdaValue=0.0, process=1))
    elif process == 2:
        leftDisparity = np.abs(disparity(finalleft, gray_right, templateSize=templateSize, window=window, lambdaValue=0.0, process=2))
        rightDisparity = np.abs(disparity(finalright, gray_left, templateSize=templateSize, window=window, lambdaValue=0.0, process=2))
    elif process == 3:
        leftDisparity = np.abs(disparity(finalleft, gray_right, templateSize=templateSize, window=window, lambdaValue=0.0, process=3))
        rightDisparity = np.abs(disparity(finalright, gray_left, templateSize=templateSize, window=window, lambdaValue=0.0, process=3))

    # Process method Outputs 
    left_map = cv2.normalize(leftDisparity, leftDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    right_map = cv2.normalize(rightDisparity, rightDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # cv2.imwrite('output/Feature/5/Left Disparity.jpg', left_map)
    # cv2.imwrite('output/Feature/5/right Disparity.jpg', right_map)
    cv2.imshow('Left Disparity SAD', left_map)
    cv2.imshow('Right Disparity SAD ', right_map)

    left_map_validate = left_map.copy()
    right_map_validate = right_map.copy()
    left_image_rows, left_image_cols = left_map_validate.shape
    for i in range(0, left_image_rows, 1):
        for j in range(0, left_image_cols, 1):
            if left_map_validate[i,j] != right_map_validate[i,j]:
                left_map_validate[i,j] = 0

    right_image_rows, right_image_cols = right_map_validate.shape
    for i in range(0, right_image_rows, 1):
        for j in range(0, right_image_cols, 1):
            if right_map_validate[i, j] != left_map_validate[i, j]:
                right_map_validate[i, j] = 0


    # cv2.imwrite('output/Feature/5/Validated Left.jpg', left_map_validate)
    # cv2.imwrite('output/Feature/5/Validated right.jpg', right_map_validate)
    cv2.imshow('left image vald', left_map_validate)
    cv2.imshow('right image vald', right_map_validate)

    # Averaging 2d Convolue
    kernel = np.ones((templateSize, templateSize), np.float32) / (templateSize^2)
    left_avg = cv2.filter2D(left_map_validate, -1, kernel)
    right_avg = cv2.filter2D(right_map_validate, -1, kernel)
    

    # cv2.imwrite('output/Feature/5/Averaged Left.jpg', left_avg)
    # cv2.imwrite('output/Feature/5/Averaged Right.jpg', right_avg)
    cv2.imshow('left avg Image', left_avg)
    cv2.imshow('Right avg Image', right_avg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()