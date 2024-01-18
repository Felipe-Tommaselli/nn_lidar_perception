import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample image of a rectangle (perspective view)
image = np.zeros((224, 224, 3), dtype=np.uint8)

# Define the coordinates of the four corners of the rectangle
rectangle_pts = np.array([[75, 50], [175, 50], [200, 200], [50, 200]], dtype=np.float32)

# Draw the rectangle on the image
cv2.polylines(image, [rectangle_pts.astype(np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)

# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("Original Image")
plt.show()

# Define the coordinates for the top-down view of the rectangle
top_down_pts = np.array([[50, 50], [200, 50], [200, 200], [50, 200]], dtype=np.float32)

# Calculate the homography matrix
homography_matrix, _ = cv2.findHomography(rectangle_pts, top_down_pts)

# Apply the homography transformation to get the top-down view
top_down_image = cv2.warpPerspective(image, homography_matrix, (400, 300))

# Display the top-down view image
plt.figure(figsize=(6, 6))
plt.imshow(top_down_image)
plt.title("Top-Down View")
plt.show()