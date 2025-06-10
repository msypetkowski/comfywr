import cv2
import numpy as np
from sklearn.cluster import KMeans


def detect_lines_lsd(img):
    """
    Detects line segments using LSD and canonicalizes segment direction prioritizing Y-axis.

    Args:
        img: 

    Returns:
        lines (list of tuples): (x1, y1, x2, y2) with dy >= 0 or if dy==0 then dx >= 0.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    detected, _, _, _ = lsd.detect(gray)
    lines = []
    if detected is not None:
        for [[x1, y1, x2, y2]] in detected:
            dx, dy = x2 - x1, y2 - y1
            # Canonicalize: ensure dy>0 or (dy==0 and dx>0)
            if (dy < 0) or (dy == 0 and dx < 0):
                x1, y1, x2, y2 = x2, y2, x1, y1
            lines.append((float(x1), float(y1), float(x2), float(y2)))
    return lines


def detect_lines_hough(img,
                       canny_thresh1=50, canny_thresh2=150,
                       rho=1, theta=np.pi/180, threshold=80,
                       min_line_length=30, max_line_gap=10):
    """
    Detects line segments using Hough and canonicalizes segment direction prioritizing Y-axis.

    Args:
        img: input image.
        canny_thresh1 (int), canny_thresh2 (int): Canny thresholds.
        rho (float), theta (float): Hough resolution.
        threshold (int), min_line_length (int), max_line_gap (int): Hough params.

    Returns:
        lines (list of tuples): (x1, y1, x2, y2) with dy >= 0 or if dy==0 then dx >= 0.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)
    raw = cv2.HoughLinesP(edges, rho=rho, theta=theta,
                          threshold=threshold,
                          minLineLength=min_line_length,
                          maxLineGap=max_line_gap)
    lines = []
    if raw is not None:
        for l in raw:
            x1, y1, x2, y2 = l[0].astype(float)
            dx, dy = x2 - x1, y2 - y1
            # Canonicalize: ensure dy>0 or (dy==0 and dx>0)
            if (dy < 0) or (dy == 0 and dx < 0):
                x1, y1, x2, y2 = x2, y2, x1, y1
            lines.append((x1, y1, x2, y2))
    return lines


def compute_line_coeffs(lines):
    """
    Given an (N,4) array of lines [(x0,y0,x1,y1)],
    returns (coeffs, coeffs_norm), each of shape (N,3),
    where coeffs[i] = (a,b,c) for a x + b y + c = 0,
    and coeffs_norm has sqrt(a²+b²)=1.
    """
    lines = np.array(lines, dtype=np.float32)
    x0, y0, x1, y1 = lines.T
    a =  y0 - y1
    b =  x1 - x0
    c =  x0 * y1 - x1 * y0
    coeffs = np.stack((a, b, c), axis=1)
    norms = np.hypot(a, b)
    coeffs_norm = coeffs / norms[:, None]
    return coeffs_norm


def cluster_lines_kmeans(lines, k=3, init_centers=None):
    """
    Clusters line segments into k groups using sklearn KMeans on normalized directions.

    Args:
        lines (list of tuples): (x1, y1, x2, y2)
        k (int): Number of clusters.
        init_centers (array-like or None): Initial cluster centers in direction space (shape kx2). If None, random init.

    Returns:
        labels (np.ndarray): Cluster labels for each line.
        centers (np.ndarray): Cluster centers as normalized direction vectors.
    """
    arr = np.array(lines, dtype=np.float32)
    dirs = arr[:, 2:4] - arr[:, 0:2]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats = dirs / norms  # normalized direction vectors

    if init_centers is not None:
        kmeans = KMeans(n_clusters=k, init=np.array(init_centers), n_init=1)
    else:
        kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(feats)
    centers = kmeans.cluster_centers_
    return labels, centers


