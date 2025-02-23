import cv2
import numpy as np
import matplotlib.pyplot as plt

def coin_detection(image):

    # Image to binary
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    # Reduction of Noise using GaussianBlur
    blurred = cv2.GaussianBlur(binary_image, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    plt.figure(figsize=(6, 6))
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection on Coins")
    plt.axis("off")
    plt.show()
    
    # Convert grayscale image to color for drawing circles
    image_color = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=100
    )
    
    # detecting circles and drawing them on the image
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Round and convert to integer
        for i in circles[0, :]:
            # Draw the outer circle (green) and the center (red)
            cv2.circle(image_color, (i[0], i[1]), i[2], (0, 0, 0), 3)
            cv2.circle(image_color, (i[0], i[1]), 2, (0, 0, 0), 3)
    
    # Detected Coins
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.title("Detected Coins using HoughCircles")
    plt.axis("off")
    plt.show()


def coin_segmentation(image):
    # grayscale image to BGR (watershed needs 3 channels)
    image_color = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    
    # Use the original grayscale image for thresholding
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological closing to remove noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # distance transform and get sure foreground areas
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    
    # Determine the sure background by dilation
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    
    # Identified unknown region (diff btw background & foreground)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label markers for the watershed algorithm
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1           # So that background is not 0 but 1
    markers[unknown == 255] = 0       # Mark the unknown region with 0
    
    # Apply the watershed algorithm on the 3-channel image
    cv2.watershed(image_color, markers)
    
    # Watershed boundaries (markers == -1)
    image_color[markers == -1] = [0, 0, 0]
    
    # Contours 
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    segmented_coins = []
    
    # Bounding box around detected coins and saved each segmented coin
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        coin = image[y:y+h, x:x+w]
        segmented_coins.append(coin)
        cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(f"segmented_coin_{idx+1}.jpg", coin)
    
    # Result plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.title("Detected Coins using Precise Region-Based Segmentation")
    plt.axis("off")
    plt.show()
    
    # Plot of each segmented coin separately
    for idx, coin in enumerate(segmented_coins):
        plt.figure(figsize=(3, 3))
        plt.imshow(coin, cmap='gray')
        plt.title(f"Segmented Coin {idx+1}")
        plt.axis("off")
        plt.show()


def count_coins(image):

    # Otsu's thresholding
    _, otsu_bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological closing for removing noise
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, kernel, iterations=12)
    
    # contours 
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_coins = len(contours)
    
    # bounding box for each detected coin
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
    
    # Coins count with bounding boxes
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Total Coins Detected: {num_coins}")
    plt.axis("off")
    plt.show()
    
    return num_coins


image = cv2.imread("coins_image.jpg", cv2.IMREAD_GRAYSCALE)

# coin detection 
coin_detection(image)

# coin segmentation 
coin_segmentation(image)

# coins count
total_coins = count_coins(image)
print(f"Total number of coins detected: {total_coins}")
