import cv2
import numpy as np

#Question 4
def translation(pts1, pts2):
    M = np.eye(3)
    diff = pts2 - pts1
    dx, dy = np.mean(diff, axis=0) 
    M[0, 2] = dx
    M[1, 2] = dy
    return M

def similarity(pts1, pts2):
    N = pts1.shape[0]
    # x2 = a*x1 - b*y1 + tx
    # y2 = b*x1 + a*y1 + ty
    # avec a = scale*cos(theta), b = scale*sin(theta)
    A = np.zeros((2*N, 4), dtype=np.float64)
    B = np.zeros((2*N),    dtype=np.float64)

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[2*i]   = [x1, -y1, 1, 0]
        A[2*i+1] = [y1,  x1, 0, 1]
        B[2*i]   = x2
        B[2*i+1] = y2

    X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, tx, ty = X

    M = np.array([
        [a, -b, tx],
        [b,  a, ty],
        [0,  0,  1]
    ], dtype=np.float64)

    return M


def similarity_cv2(pts_src, pts_dst):
    """
    Estimate a SIMILARITY transform (4 DOF: scale, rotation, tx, ty)
    using cv2.estimateAffinePartial2D with built-in RANSAC.
 
    cv2.estimateAffinePartial2D retourne une matrice 2x3 :
        [[a, -b, tx],
         [b,  a, ty]]
    On ajoute [0, 0, 1] pour obtenir une 3x3 homogene compatible
    avec warpPerspective et notre ransac_filter.
 
    Pourquoi cv2 plutot que notre version maison ?
    - RANSAC integre en C++ → robuste aux outliers
    - Impose la contrainte similarity stricte (a^2 + b^2 = s^2)
      que lstsq ne garantit pas avec des points bruites
    """
    import cv2
    if len(pts_src) < 2:
        raise ValueError("similarity requires at least 2 point pairs")
 
    M, inliers = cv2.estimateAffinePartial2D(
        pts_src, pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.999
    )
    if M is None:
        raise np.linalg.LinAlgError("estimateAffinePartial2D returned None")
 
    # 2x3 → 3x3
    H = np.vstack([M, [0, 0, 1]]).astype(np.float64)
    return H

def affine(pts1, pts2):
    N = pts1.shape[0] 
    # We're going to reproduce the equation and solve it to get the parameters
    A = np.zeros((2 * N, 6), dtype=np.float32)
    B = np.zeros((2 * N, 1), dtype=np.float32)
    
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[2 * i]     = [1, 0,  x1, y1, 0, 0]
        B[2 * i]     = [x2 - x1] 
        
        A[2 * i + 1] = [0, 1,  0, 0, x1, y1]
        B[2 * i + 1] = [y2 - y1]
    
    X, _,_,_ = np.linalg.lstsq(A, B, rcond=None)
    tx, ty, a00, a01, a10, a11 = X[0, 0], X[1, 0], X[2, 0], X[3, 0], X[4, 0], X[5, 0]

    M = np.array([
        [ 1 + a00,  a01,  tx],
        [ a10,   1+ a11,  ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return M

def homography(pts1, pts2):
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H


def compute_reprojection_error(H, pts_src, pts_dst):
    """Compute mean reprojection error for a given homography H."""
    n = len(pts_src)
    src_h = np.c_[pts_src, np.ones(n)].T          # 3 x n
    projected = (H @ src_h).T                       # n x 3
    projected /= projected[:, [2]]                  # normalise
    error = np.linalg.norm(projected[:, :2] - pts_dst, axis=1)
    return float(np.mean(error))
