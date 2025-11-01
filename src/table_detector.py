import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from config import TABLE_DETECTION_SCALE, JOINT_TOLERANCE, CELL_PADDING


def extract_table_mask(gray, scale=TABLE_DETECTION_SCALE):
    """Extract horizontal and vertical lines to create table mask"""
    # Adaptive binarization with inversion
    bin_img = cv2.adaptiveThreshold(
        ~gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 15, -2
    )
    
    # Detect horizontal lines
    horizontal = bin_img.copy()
    cols = horizontal.shape[1]
    horizontalsize = max(1, cols // scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, iterations=1)
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=1)
    
    # Detect vertical lines
    vertical = bin_img.copy()
    rows = vertical.shape[0]
    verticalsize = max(1, rows // scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, iterations=1)
    vertical = cv2.dilate(vertical, verticalStructure, iterations=1)
    
    # Create mask and detect joints
    mask = cv2.add(horizontal, vertical)
    joints = cv2.bitwise_and(horizontal, vertical)
    
    return mask, joints, horizontal, vertical


def detect_largest_table(mask):
    """Find the largest table region in the mask"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return None
    
    # Sort by area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Find first contour that looks like a table
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > 5000 and w > 50 and h > 50:
            return x, y, w, h
    
    # Fallback to largest
    x, y, w, h = cv2.boundingRect(cnts[0])
    return x, y, w, h


def joints_to_grid(joints_img, joint_tol=JOINT_TOLERANCE):
    """Convert joint points to grid coordinates"""
    # Find joint centroids
    cnts, _ = cv2.findContours(joints_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    pts = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        cy = y + h // 2
        pts.append((cx, cy))
    
    if len(pts) == 0:
        return None, None
    
    pts = np.array(pts)
    
    # Cluster X and Y coordinates separately using DBSCAN
    xs = pts[:, 0].reshape(-1, 1).astype(float)
    ys = pts[:, 1].reshape(-1, 1).astype(float)
    
    clust_x = DBSCAN(eps=joint_tol, min_samples=1).fit(xs)
    clust_y = DBSCAN(eps=joint_tol, min_samples=1).fit(ys)
    
    # Get unique X coordinates
    unique_x = []
    for lab in sorted(set(clust_x.labels_)):
        unique_x.append(int(np.mean(xs[clust_x.labels_ == lab])))
    
    # Get unique Y coordinates
    unique_y = []
    for lab in sorted(set(clust_y.labels_)):
        unique_y.append(int(np.mean(ys[clust_y.labels_ == lab])))
    
    unique_x = sorted(unique_x)
    unique_y = sorted(unique_y)
    
    return unique_x, unique_y


def crop_cells_from_grid(table_roi, unique_x, unique_y, pad=CELL_PADDING):
    """Extract individual cells from the table based on grid"""
    cells = []
    
    for i in range(len(unique_y) - 1):
        row_cells = []
        y1 = unique_y[i]
        y2 = unique_y[i + 1]
        
        for j in range(len(unique_x) - 1):
            x1 = unique_x[j]
            x2 = unique_x[j + 1]
            
            # Add padding but stay within bounds
            H, W = table_roi.shape[:2]
            xa = max(0, x1 - pad)
            ya = max(0, y1 - pad)
            xb = min(W, x2 + pad)
            yb = min(H, y2 + pad)
            
            if xb - xa > 3 and yb - ya > 3:
                cell_img = table_roi[ya:yb, xa:xb]
                row_cells.append(((xa, ya, xb-xa, yb-ya), cell_img))
            else:
                row_cells.append(((xa, ya, xb-xa, yb-ya), None))
        
        cells.append(row_cells)
    
    return cells


def detect_table_structure(img, scale=TABLE_DETECTION_SCALE, joint_tol=JOINT_TOLERANCE):
    """Main function to detect table structure and extract cells"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask, joints_img, horizontal, vertical = extract_table_mask(gray, scale=scale)
    
    # Find table region
    tbl = detect_largest_table(mask)
    if tbl is None:
        raise RuntimeError("No table detected. Try adjusting scale parameter.")
    
    x, y, w, h = tbl
    table_roi = img[y:y+h, x:x+w]
    
    # Extract joints in ROI
    _, joints_roi, _, _ = extract_table_mask(cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY), scale=scale)
    
    # Find joint centroids
    cnts, _ = cv2.findContours(joints_roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        pts.append((bx + bw//2, by + bh//2))
    
    # Get grid coordinates
    if len(pts) < 4:
        # Fallback: use line projections
        unique_x, unique_y = get_grid_from_projections(table_roi, scale)
    else:
        unique_x, unique_y = joints_to_grid(joints_roi, joint_tol=joint_tol)
    
    # Ensure borders are included
    if unique_x and unique_y:
        if unique_x[0] > 5:
            unique_x = [0] + unique_x
        if unique_x[-1] < table_roi.shape[1] - 5:
            unique_x = unique_x + [table_roi.shape[1] - 1]
        if unique_y[0] > 5:
            unique_y = [0] + unique_y
        if unique_y[-1] < table_roi.shape[0] - 5:
            unique_y = unique_y + [table_roi.shape[0] - 1]
    
    if unique_x is None or unique_y is None or len(unique_x) < 2 or len(unique_y) < 2:
        raise RuntimeError("Failed to compute grid. Try adjusting parameters.")
    
    # Extract cells
    cells = crop_cells_from_grid(table_roi, unique_x, unique_y)
    
    return cells, table_roi


def get_grid_from_projections(table_roi, scale):
    """Fallback method to get grid using line projections"""
    gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    
    hmask = cv2.erode(bin_img, cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, table_roi.shape[1]//scale), 1)))
    vmask = cv2.erode(bin_img, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, table_roi.shape[0]//scale))))
    
    proj_x = np.sum(vmask, axis=0)
    proj_y = np.sum(hmask, axis=1)
    
    cols = np.where(proj_x > np.max(proj_x) * 0.5)[0]
    rows = np.where(proj_y > np.max(proj_y) * 0.5)[0]
    
    def runs_to_centers(indices):
        if len(indices) == 0:
            return []
        groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        centers = [int(np.mean(g)) for g in groups]
        return centers
    
    unique_x = runs_to_centers(cols)
    unique_y = runs_to_centers(rows)
    
    # Include borders
    unique_x = [0] + unique_x + [table_roi.shape[1] - 1]
    unique_y = [0] + unique_y + [table_roi.shape[0] - 1]
    
    return unique_x, unique_y
