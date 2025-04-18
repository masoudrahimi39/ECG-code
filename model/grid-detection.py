import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_dominant_lines(ys):
    ys = sorted(ys)
    final_ys = []
    group = []
    last_y = -100
    for y in [*ys, 10_0000]:
        if np.abs(y-last_y)>20:
            if len(group)>1:
                final_ys.append(np.median(group))
            group = [y]
        else:
            group.append(y)
        last_y = y
    return final_ys



def find_gridsize(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # edges = cv2.Canny(blurred_image, 40, 150)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # image = clahe.apply(image)

    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    n_clusters = 3
    attempts = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    while attempts > 0:
        kmeans = KMeans(n_clusters=n_clusters, random_state=attempts, n_init = 'auto').fit(pixel_values)
        centers = np.sort(kmeans.cluster_centers_.flatten())
        
        min_distance = np.min(np.diff(centers))
        
        if min_distance > 30:
            break
        else:
            attempts -= 1
            if attempts == 0:
                raise ValueError("Clusters are not well separated. Try a different image or adjust parameters.")

    labels = kmeans.labels_.flatten()

    closest_cluster_index = np.argsort(centers)[1]

    mask = np.isin(labels, closest_cluster_index)
    binary_image = np.zeros_like(image)
    binary_image[mask.reshape(image.shape)] = 255

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=5)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,2))
    horizontal_lines = cv2.morphologyEx(closing, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
    vertical_lines = cv2.morphologyEx(closing, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # original_image = cv2.imread(image_path)
    # original_image_copy = original_image.copy()

    lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=100, minLineLength=30, maxLineGap=10)
    if lines is not None:
        ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ys.append((y1+y2)/2)
        ys = find_dominant_lines(ys)
        # for y in ys:
        #     y = np.uint(y)
        #     cv2.line(original_image_copy, (0, y), (original_image_copy.shape[1], y), (0, 255, 0), 2)

    lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=100, minLineLength=30, maxLineGap=10)
    if lines is not None:
        xs = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            xs.append((x1+x2)/2)
        xs = find_dominant_lines(xs)
        # for x in xs:
        #     x = np.uint(x)
        #     cv2.line(original_image_copy, (x, 0), (x, original_image_copy.shape[0]), (0, 255, 0), 2)
    
    xs = np.array(xs)
    ys = np.array(ys)

    dist_y = 0
    dist_x = 0

    if len(xs)>2:
        dist_x = np.median(xs[1:]-xs[:-1])
    if len(ys)>2:
        dist_y = np.median(ys[1:]-ys[:-1])

    if dist_x==0:
        dist_x = dist_y
    if dist_y==0:
        dist_y = dist_x

    return dist_x, dist_y
    
if __name__ == "__main__":
    image_path = "datasets/image_dataset_v8.0/segmentation/train/images/1_3by4_lead_1.png"

    dist_x, dist_y = find_gridsize(image_path)
