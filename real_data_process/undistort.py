import cv2
import numpy as np


def undistort_image(img: np.ndarray):
    h, w = img.shape[:2]
    mtx = np.array(
        [[402.60350228,   0.,         263.30000918],
         [0.,       537.76023089, 278.24728515],
            [0.,        0.,      1.]]
    )
    dist_coeffs = np.array(
        [[-0.31085325, -0.11558236,  0.00249467, -0.00088277,  0.51442531]])

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist_coeffs, (w, h), 1, (w, h))

    undistorted_img = cv2.undistort(img, newcameramtx, dist_coeffs)
    x, y, w, h = roi
    dst = undistorted_img[y:y+h, x:x+w]
    return dst
