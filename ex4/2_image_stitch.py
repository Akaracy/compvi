import cv2
import numpy as np
import glob
import os
from module.transformation import translation, similarity, affine, homography, compute_reprojection_error, similarity_cv2

# ── helpers ──────────────────────────────────────────────────────────────────

def load_images(pattern="stitch_pic/*.jpg"):
    paths = sorted(glob.glob(pattern))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least 2 images matching '{pattern}'")
    imgs = [cv2.imread(p) for p in paths]
    print(f"Loaded {len(imgs)} images from {pattern}")
    return imgs


def detect_and_match(img1, img2, ratio=0.75):
    """SIFT + FLANN matching with Lowe ratio test."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    raw = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2

#fixed ransac
def ransac_filter(H_func, pts_src, pts_dst, threshold=3.0, iterations=2000, seed=42):
    """
    RANSAC with fixed seed for reproducibility.
 
    Why the result changed between runs
    ------------------------------------
    RANSAC is random — it samples point subsets randomly at each iteration.
    Without a fixed seed, np.random gives different sequences each run.
    If the random draw happens to pick outlier points early and never recovers
    a good consensus set, the final H is wrong → ghost/double-exposure result.
 
    Fixes applied
    -------------
    1. seed=42         : np.random.seed() makes every run identical
    2. threshold=3.0   : tighter inlier threshold (was 4.0) → fewer false inliers
    3. iterations=2000 : more attempts → less chance of missing the good solution
    4. Early exit      : stop if >80% of points are inliers (clearly converged)
    """
    np.random.seed(seed)  # ← reproducibility fix
 
    min_pts = {"translation": 1, "similarity": 2, "affine": 3, "homography": 4}
    n_min = min_pts.get(H_func.__name__, 4)
    n = len(pts_src)
 
    best_H, best_inliers = None, []
 
    for _ in range(iterations):
        idx = np.random.choice(n, n_min, replace=False)
        try:
            H = H_func(pts_src[idx], pts_dst[idx])
        except (np.linalg.LinAlgError, ValueError):
            continue
 
        src_h = np.c_[pts_src, np.ones(n)].T
        proj  = (H @ src_h).T
        proj /= proj[:, [2]]
        errors  = np.linalg.norm(proj[:, :2] - pts_dst, axis=1)
        inliers = np.where(errors < threshold)[0]
 
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
            # Early exit: clearly a good solution
            if len(best_inliers) > 0.8 * n:
                break
 
    # Re-estimate on all inliers
    if len(best_inliers) >= n_min:
        best_H = H_func(pts_src[best_inliers], pts_dst[best_inliers])
 
    mask = np.zeros(n, dtype=bool)
    mask[best_inliers] = True
    print(f"  RANSAC → {mask.sum()}/{n} inliers")
    return best_H, mask
 


def warp_and_stitch(img1, img2, H):
    """
    Warp img1 onto img2's plane using homography H (maps img1→img2),
    then blend side by side.  Handles negative offsets with a canvas shift.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Corners of img1 in img2's space
    corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    corners1_t = cv2.perspectiveTransform(corners1, H)

    # All corners together
    corners2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    all_corners = np.concatenate([corners1_t, corners2], axis=0)

    xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel()) + 1

    # Translation to keep everything in positive coordinates
    shift = np.array([[1, 0, -xmin],
                      [0, 1, -ymin],
                      [0, 0,      1]], dtype=np.float64)

    canvas_w = xmax - xmin
    canvas_h = ymax - ymin

    # Warp img1
    warped1 = cv2.warpPerspective(img1, shift @ H, (canvas_w, canvas_h))

    # Place img2 on canvas
    canvas = warped1.copy()
    y_off, x_off = -ymin, -xmin
    # Simple copy (img2 is the "anchor")
    roi = canvas[y_off:y_off + h2, x_off:x_off + w2]
    # Only overwrite where warped1 is dark (simple blending)
    mask_img2 = np.any(warped1[y_off:y_off + h2, x_off:x_off + w2] == 0, axis=2)
    roi[mask_img2]  = img2[mask_img2]
    roi[~mask_img2] = cv2.addWeighted(
        roi[~mask_img2], 0.5,
        img2[~mask_img2], 0.5, 0
    )
    canvas[y_off:y_off + h2, x_off:x_off + w2] = roi

    return canvas


def display_fit(title, img, max_w=1200, max_h=700):
    """Show image rescaled to fit the screen — no more giant windows."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)          # never upscale
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)


def save(path, img):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    cv2.imwrite(path, img)
    print(f"  Saved → {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    images = load_images("stitch_pic/*.jpg")

    img1, img2 = images[0], images[1]
    print("\n── Feature matching (img0 ↔ img1) ──")
    pts1, pts2 = detect_and_match(img1, img2)

    transforms = [
        #("Translation", translation),
        ("Similarity",  similarity),
        #("Affine",      affine),
        #("Homography",  homography),
    ]

    results = {}

    for name, fn in transforms:
        print(f"\n── {name} ──")
        #H, mask = ransac_filter(fn, pts1, pts2, seed = 342)
        H = similarity_cv2(pts1, pts2)
        # err  = compute_reprojection_error(H, pts1[mask], pts2[mask])
        # print(f"  Inliers : {n_in}/{len(pts1)}")
        # print(f"  Mean reprojection error: {err:.2f} px")
        print(f"  H =\n{np.round(H, 4)}")

        stitched = warp_and_stitch(img1, img2, H)
        results[name] = (stitched)

        out_path = f"output1_{name.lower()}.jpg"
        save(out_path, stitched)
        display_fit(f"{name} stitch", stitched)

    # Summary
    # print("\n── Summary ──")
    # print(f"{'Transform':<14} {'Inliers':>8} {'Error (px)':>12}")
    # print("-" * 36)
    # for name, (_, err, n_in) in results.items():
    #     print(f"{name:<14} {n_in:>8} {err:>12.2f}")

    print("\nPress any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
