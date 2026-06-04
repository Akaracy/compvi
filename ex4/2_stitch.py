import cv2
import numpy as np
import glob
import os
from module.transformation import translation, similarity, affine, homography, compute_reprojection_error


def load_images(pattern="stitch_pic/*.jpg"):
    paths = sorted(glob.glob(pattern))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least 2 images matching '{pattern}'")
    imgs = [cv2.imread(p) for p in paths]
    print(f"Loaded {len(imgs)} images from {pattern}")
    return imgs


def detect_and_match(img1, img2, ratio=0.75):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw  = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    print(f"  {len(good)} good matches")
    return pts1, pts2


def ransac_filter(H_func, pts_src, pts_dst, threshold=3.0, iterations=2000):
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
            if len(best_inliers) > 0.8 * n:
                break

    if len(best_inliers) >= n_min:
        best_H = H_func(pts_src[best_inliers], pts_dst[best_inliers])

    mask = np.zeros(n, dtype=bool)
    mask[best_inliers] = True
    print(f"  RANSAC → {mask.sum()}/{n} inliers")
    return best_H, mask


def warp_and_stitch(img1, img2, H):
    """
    Warp img1 → plan de img2.

    Stratégie de blend
    ------------------
    On définit 3 zones sur le canvas :
      A : pixels couverts par img1 warpée SEULEMENT  → img1 warpée
      B : pixels couverts par img2 SEULEMENT         → img2
      C : chevauchement (les deux valides)            → img2 prioritaire (net, pas de fantôme)

    Pourquoi img2 prioritaire dans C ?
    img2 est l'image de référence (non transformée) → pas de dégradation
    due au warp. Prendre img2 dans la zone de overlap donne l'image la plus nette.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1   = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    corners1_t = cv2.perspectiveTransform(corners1, H)
    corners2   = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    all_c      = np.concatenate([corners1_t, corners2], axis=0)

    xmin, ymin = np.int32(all_c.min(axis=0).ravel())
    xmax, ymax = np.int32(all_c.max(axis=0).ravel()) + 1
    shift    = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float64)
    canvas_w, canvas_h = xmax - xmin, ymax - ymin

    # Warp img1
    warped = cv2.warpPerspective(img1, shift @ H, (canvas_w, canvas_h))
    mask_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0   # pixels valides de img1

    # Placer img2 sur un canvas vide
    canvas_img2 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    y0, x0 = -ymin, -xmin
    canvas_img2[y0:y0+h2, x0:x0+w2] = img2
    mask_img2 = np.zeros((canvas_h, canvas_w), dtype=bool)
    mask_img2[y0:y0+h2, x0:x0+w2] = True                        # pixels valides de img2

    # Zones
    only_warped = mask_warped & ~mask_img2   # A : img1 seule
    only_img2   = mask_img2 & ~mask_warped   # B : img2 seule
    overlap     = mask_warped & mask_img2    # C : chevauchement

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[only_warped] = warped[only_warped]          # A
    canvas[only_img2]   = canvas_img2[only_img2]       # B
    canvas[overlap]     = canvas_img2[overlap]         # C → img2 prioritaire, image nette

    return canvas


def display_fit(title, img, max_w=1200, max_h=700):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)


def save(path, img):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    cv2.imwrite(path, img)
    print(f"  Saved → {path}")


def main():
    images = load_images("stitch_pic/*.jpg")
    img1, img2 = images[0], images[1]

    print("\n── Feature matching (img0 ↔ img1) ──")
    pts1, pts2 = detect_and_match(img1, img2)

    transforms = [
        ("Translation", translation),
        ("Similarity",  similarity),
        ("Affine",      affine),
        ("Homography",  homography),
    ]

    results = {}
    for name, fn in transforms:
        print(f"\n── {name} ──")
        H, mask = ransac_filter(fn, pts1, pts2)
        n_in = mask.sum()
        err  = compute_reprojection_error(H, pts1[mask], pts2[mask])
        print(f"  Inliers : {n_in}/{len(pts1)}")
        print(f"  Mean reprojection error: {err:.2f} px")

        stitched = warp_and_stitch(img1, img2, H)
        results[name] = (stitched, err, n_in)
        save(f"output2_{name.lower()}.jpg", stitched)
        display_fit(f"{name} stitch", stitched)

    print("\n── Summary ──")
    print(f"{'Transform':<14} {'Inliers':>8} {'Error (px)':>12}")
    print("-" * 36)
    for name, (_, err, n_in) in results.items():
        print(f"{name:<14} {n_in:>8} {err:>12.2f}")

    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
