import cv2
import numpy as np

def digitize_binary_image(img, mv_per_pixel, ms_per_pixel):
    # Find contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract all x, y points from the largest contour
    contour_points = [(point[0][0], point[0][1]) for point in largest_contour]

    # Create a dictionary to hold all the y-values for each x
    x_to_y_values = {}

    for x, y in contour_points:
        if x not in x_to_y_values:
            x_to_y_values[x] = []
        x_to_y_values[x].append(y)

    # For each x, find the median of the corresponding y-values
    extracted_function = []
    for x in sorted(x_to_y_values.keys()):
        y_median = np.median(x_to_y_values[x])
        extracted_function.append((x, int(y_median)))

    # Plot the extracted function
    x_vals, y_vals = zip(*extracted_function)

    x_vals -= np.min(x_vals)
    y_vals -= np.median(y_vals)

    return x_vals/ms_per_pixel, -y_vals/mv_per_pixel
