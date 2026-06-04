"""
similarity_stitch.py
--------------------
Self-contained script: similarity transform (homemade + cv2) + stitch on 2 images.
No external transformation.py needed.
"""

import cv2
import numpy as np
import glob


# ─────────────────────────────────────────────────────────────
# SIMILARITY — 2 versions
# ─────────────────────────────────────────────────────────────

def similarity_homemade(pts_src, pts_dst):
    """
    Similarity (4 DOF) from scratch via least squares.
    Model: x' = a*x - b*y + tx
           y' = b*x + a*y + ty
    Builds a linear system A·[a, b, tx, ty]^T = rhs and solves with lstsq.
    Does NOT include RANSAC — sensitive to outliers.
    """
    n = len(pts_src)
    A   = np.zeros((2 * n, 4), dtype=np.float64)
    rhs = np.zeros(2 * n,      dtype=np.float64)

    for i, ((x, y), (xp, yp)) in enumerate(zip(pts_src, pts_dst)):
        A[2*i]     = [ x, -y, 1, 0]
        rhs[2*i]   = xp
        A[2*i + 1] = [ y,  x, 0, 1]
        rhs[2*i+1] = yp

    params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, tx, ty = params
    return np.array([[ a, -b, tx],
                     [ b,  a, ty],
                     [ 0,  0,  1]], dtype=np.float64)


def similarity_cv2(pts_src, pts_dst):
    """
    Similarity (4 DOF) using cv2.estimateAffinePartial2D with built-in RANSAC.
    Returns a 3x3 homogeneous matrix (adds [0,0,1] row to the 2x3 cv2 output).
    Robust to outliers — recommended over the homemade version.
    """
    M, _ = cv2.estimateAffinePartial2D(
        pts_src, pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.999
    )
    if M is None:
        raise RuntimeError("estimateAffinePartial2D failed — not enough matches?")
    return np.vstack([M, [0, 0, 1]]).astype(np.float64)


# ─────────────────────────────────────────────────────────────
# RANSAC wrapper (used around homemade only)
# ─────────────────────────────────────────────────────────────

def ransac_filter(H_func, pts_src, pts_dst, threshold=3.0, iterations=2000):
    """
    Generic RANSAC for any transform function.
    Draws the minimum number of points, estimates H, counts inliers, repeats.
    Only needed for homemade similarity (cv2 version has RANSAC built in).
    """
    n_min = 2   # minimum for similarity
    n     = len(pts_src)
    best_H, best_inliers = None, []

    for _ in range(iterations):
        idx = np.random.choice(n, n_min, replace=False)
        try:
            H = H_func(pts_src[idx], pts_dst[idx])
        except Exception:
            continue

        src_h   = np.c_[pts_src, np.ones(n)].T
        proj    = (H @ src_h).T
        proj   /= proj[:, [2]]
        errors  = np.linalg.norm(proj[:, :2] - pts_dst, axis=1)
        inliers = np.where(errors < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
            if len(best_inliers) > 0.8 * n:
                break

    if len(best_inliers) >= n_min:
        best_H = H_func(pts_src[best_inliers], pts_dst[best_inliers])

    mask = np.zeros(n, dtype=bool)
    mask[best_inliers] = True
    return best_H, mask


# ─────────────────────────────────────────────────────────────
# FEATURE MATCHING
# ─────────────────────────────────────────────────────────────

def detect_and_match(img1, img2, ratio=0.75):
    sift    = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw if m.distance < ratio * n.distance]
    pts1    = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2    = np.float32([kp2[m.trainIdx].pt for m in good])
    print(f"  {len(good)} good matches found")
    return pts1, pts2, kp1, kp2, good


# ─────────────────────────────────────────────────────────────
# DRAW MATCHES
# ─────────────────────────────────────────────────────────────

def draw_matches(img1, img2, kp1, kp2, good, mask=None, title="Matches"):
    if mask is not None:
        p_in  = dict(matchColor=(0,255,0), singlePointColor=None,
                     matchesMask=mask.astype(int).tolist(),
                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        p_out = dict(matchColor=(0,0,255), singlePointColor=None,
                     matchesMask=(~mask).astype(int).tolist(),
                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None,  **p_in)
        vis = cv2.drawMatches(img1, kp1, img2, kp2, good, vis,   **p_out)
    else:
        vis = cv2.drawMatches(img1, kp1, img2, kp2, good[:60], None,
                              matchColor=(0,255,0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    h, w    = vis.shape[:2]
    scale   = min(1400/w, 600/h, 1.0)
    vis     = cv2.resize(vis, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, vis)
    cv2.imwrite(f"outputSim_{title.replace(' ','_')}.jpg", vis)


# ─────────────────────────────────────────────────────────────
# STITCH
# ─────────────────────────────────────────────────────────────

def stitch(img_src, img_ref, H):
    """
    Warp img_src into img_ref's plane.
 
    Blend strategy
    --------------
    - Only img_src  (A) → img_src pixels
    - Only img_ref  (B) → img_ref pixels
    - Overlap       (C) → linear gradient across the overlap width:
          left edge  of overlap → 100 % img_src (warped)
          right edge of overlap → 100 % img_ref
      This removes the hard seam without creating a ghost effect,
      because the images are already well-aligned by the transform.
    """
    h1, w1 = img_src.shape[:2]
    h2, w2 = img_ref.shape[:2]
 
    c1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    c2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    all_c = np.concatenate([cv2.perspectiveTransform(c1, H), c2], axis=0)
 
    xmin, ymin = np.int32(all_c.min(axis=0).ravel())
    xmax, ymax = np.int32(all_c.max(axis=0).ravel()) + 1
    shift  = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float64)
    cw, ch = xmax-xmin, ymax-ymin
 
    warped     = cv2.warpPerspective(img_src, shift@H, (cw,ch))
    mask_w     = cv2.cvtColor(warped,  cv2.COLOR_BGR2GRAY) > 0
 
    ref_canvas = np.zeros((ch,cw,3), dtype=np.uint8)
    y0, x0 = -ymin, -xmin
    ref_canvas[y0:y0+h2, x0:x0+w2] = img_ref
    mask_r = np.zeros((ch,cw), dtype=bool)
    mask_r[y0:y0+h2, x0:x0+w2] = True
 
    overlap = mask_w & mask_r
 
    # Build per-pixel alpha for the overlap zone
    # alpha=0 → use warped, alpha=1 → use ref
    # Gradient along x: from left edge of overlap to right edge
    alpha = np.zeros((ch,cw), dtype=np.float32)
    if overlap.any():
        cols_with_overlap = np.where(overlap.any(axis=0))[0]
        x_left  = cols_with_overlap[0]
        x_right = cols_with_overlap[-1]
        span = max(x_right - x_left, 1)
        for x in cols_with_overlap:
            alpha[overlap[:,x], x] = (x - x_left) / span
 
    alpha3 = alpha[:,:,np.newaxis]
 
    result = np.zeros((ch,cw,3), dtype=np.uint8)
    # A — only warped
    only_w = mask_w & ~mask_r
    result[only_w] = warped[only_w]
    # B — only ref
    only_r = mask_r & ~mask_w
    result[only_r] = ref_canvas[only_r]
    # C — gradient blend
    blended = ((1-alpha3)*warped.astype(np.float32)
               + alpha3*ref_canvas.astype(np.float32)).astype(np.uint8)
    result[overlap] = blended[overlap]
 
    return result



def reprojection_error(H, pts_src, pts_dst):
    n     = len(pts_src)
    src_h = np.c_[pts_src, np.ones(n)].T
    proj  = (H @ src_h).T
    proj /= proj[:, [2]]
    return float(np.mean(np.linalg.norm(proj[:, :2] - pts_dst, axis=1)))


def show(title, img, max_w=1300, max_h=750):
    h, w  = img.shape[:2]
    scale = min(max_w/w, max_h/h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)
    cv2.imwrite(f"output_{title.replace(' ','_')}.jpg", img)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    paths = sorted(glob.glob("stitch_pic/*.jpg"))
    if len(paths) < 2:
        raise FileNotFoundError("Need at least 2 images in stitch_pic/")
    img1 = cv2.imread(paths[1])
    img2 = cv2.imread(paths[2])
    print(f"Images: {paths[0]}  {paths[1]}")

    print("\n── Matching ──")
    pts1, pts2, kp1, kp2, good = detect_and_match(img1, img2)

    # ── Homemade similarity + our RANSAC ─────────────────────
    print("\n── Homemade similarity + RANSAC ──")
    H_hm, mask_hm = ransac_filter(similarity_homemade, pts1, pts2)
    err_hm = reprojection_error(H_hm, pts1[mask_hm], pts2[mask_hm])
    print(f"  Inliers : {mask_hm.sum()}/{len(pts1)}")
    print(f"  Error   : {err_hm:.2f} px")
    print(f"  H =\n{np.round(H_hm, 4)}")
    # draw_matches(img1, img2, kp1, kp2, good, mask=mask_hm,
    #              title="matches homemade")
    show("stitch homemade", stitch(img1, img2, H_hm))

    # ── cv2 similarity (RANSAC built-in) ─────────────────────
    print("\n── cv2 similarity (RANSAC built-in) ──")
    H_cv, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    mask_cv2_full, _ = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.999)
    H_cv2 = similarity_cv2(pts1, pts2)
    # compute inlier mask for display
    src_h  = np.c_[pts1, np.ones(len(pts1))].T
    proj   = (H_cv2 @ src_h).T;  proj /= proj[:, [2]]
    mask_c = np.linalg.norm(proj[:, :2] - pts2, axis=1) < 3.0
    err_cv = reprojection_error(H_cv2, pts1[mask_c], pts2[mask_c])
    print(f"  Inliers : {mask_c.sum()}/{len(pts1)}")
    print(f"  Error   : {err_cv:.2f} px")
    print(f"  H =\n{np.round(H_cv2, 4)}")
    # draw_matches(img1, img2, kp1, kp2, good, mask=mask_c,
    #              title="matches cv2")
    show("stitch cv2", stitch(img1, img2, H_cv2))

    # ── Summary ──────────────────────────────────────────────
    print("\n── Summary ──")
    print(f"{'Method':<25} {'Inliers':>8} {'Error (px)':>12}")
    print("-" * 47)
    print(f"{'Homemade + RANSAC':<25} {mask_hm.sum():>8} {err_hm:>12.2f}")
    print(f"{'cv2 estimateAffinePartial':<25} {mask_c.sum():>8} {err_cv:>12.2f}")

    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
