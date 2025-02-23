import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resizing the image
def resize_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# Find keypoints using SIFT detector
def keypoints_SIFT(gray_img, ori_img, a):
    sift = cv2.SIFT_create()
    kp_desc = sift.detectAndCompute(gray_img, None)
    
    pano_kp = cv2.drawKeypoints(ori_img, kp_desc[0], None, color=(0, 255, 0),flags=cv2.DrawMatchesFlags_DEFAULT)
    # plt.figure(figsize=(15, 5))
    plt.title(f"{a} key_points")
    plt.imshow(cv2.cvtColor(pano_kp, cv2.COLOR_BGR2RGB))
    plt.show()

    return kp_desc

def estimate_homography(img1, img2, kp_desc1, kp_desc2):
    kp1, desc1 = kp_desc1
    kp2, desc2 = kp_desc2

    index_params = dict(algorithm=0, trees=15)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matched = flann.match(desc1, desc2)
    sorted_match = sorted(matched, key=lambda x: x.distance)[:50]

    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, sorted_match, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 8))
    plt.title("Matches")
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    src_pts = np.float32([kp1[m.queryIdx].pt for m in sorted_match]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in sorted_match]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return H

def warp_corners(img, H):
    h, w = img.shape[:2]
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H)
    return warped

def stitch_three_images(left_img, ref_img, right_img):

    # Resizing the image
    left_img = resize_img(left_img, 50)
    ref_img  = resize_img(ref_img, 50)
    right_img = resize_img(right_img, 50)

    # Convert to grayscale for homography estimation
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Compute keypoints for all
    kp_left = keypoints_SIFT(gray_left, left_img,"left_img")
    kp_ref = keypoints_SIFT(gray_ref, ref_img,"ref_img")
    kp_right = keypoints_SIFT(gray_right, right_img,"right_img")
   
    # Compute homography from left to ref image and right to ref image
    
    homo_left = estimate_homography(left_img, ref_img, kp_left, kp_ref)
    homo_right = estimate_homography(right_img, ref_img, kp_right, kp_ref)

    # Compute corners for each image in the reference coordinate space
    h_ref, w_ref = ref_img.shape[:2]
    ref_corners = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32).reshape(-1, 1, 2)

    left_corners = warp_corners(left_img, homo_left)
    right_corners = warp_corners(right_img, homo_right)

    # Get overall bounds
    all_corners = np.concatenate((ref_corners, left_corners, right_corners), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Compute translation matrix to ensure all coordinates are positive
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    # Warp left and right images into the reference frame with translation
    warped_left = cv2.warpPerspective(left_img, translation.dot(homo_left), (canvas_width, canvas_height))
    warped_right = cv2.warpPerspective(right_img, translation.dot(homo_right), (canvas_width, canvas_height))

    # Place the reference image on the canvas
    panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    # Reference image position after translation (its top-left corner moves by (-x_min, -y_min))
    panorama[-y_min:h_ref - y_min, -x_min:w_ref - x_min] = ref_img

    # Overlay the warped left and right images (simply replace non-black pixels)
    mask_left = (warped_left > 0)
    panorama[mask_left] = warped_left[mask_left]

    mask_right = (warped_right > 0)
    panorama[mask_right] = warped_right[mask_right]

    return panorama

# Read and resize images
left_img = cv2.imread('pano1.jpg')   
ref_img  = cv2.imread('pano2.jpg')
right_img = cv2.imread('pano3.jpg')  

# Stitch the three images
panorama = stitch_three_images(left_img, ref_img, right_img)

if panorama is not None:
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("panorama-1.jpg", panorama)
