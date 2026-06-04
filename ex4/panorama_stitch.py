"""
panorama_stitch.py
------------------
- draw_matches()      : visualise les correspondances entre 2 images
- ransac_filter()     : filtre les outliers pour n'importe quelle transform
- stitch_two()        : warp + blend 2 images
- stitch_panorama()   : chaîne les 6 images en composant les homographies
"""

import cv2
import numpy as np
import glob
import os
from module.transformation import translation, similarity, affine, homography, compute_reprojection_error
from module.display_helper import mosaic_maker, load_images, show, mosaic_maker_titles
from module.match import draw_matches, ransac_filter, detect_and_match

def _normalize_H(H, pts):
    """
    Normalise H pour éviter la dérive numérique lors de la composition.
    Divise par H[2,2] pour s'assurer que le coin inférieur droit = 1.
    """
    return H / H[2, 2]


# ─────────────────────────────────────────────────────────────
# WARP + BLEND (2 images)
# ─────────────────────────────────────────────────────────────

def stitch_two(img_src, img_ref, H):
    """
    Warp img_src vers le plan de img_ref via H (H : img_src → img_ref).
    Étend le canvas pour accueillir les deux images, blend simple.
    """
    h_src, w_src = img_src.shape[:2]
    h_ref, w_ref = img_ref.shape[:2]

    # Coins de img_src projetés dans le plan de img_ref
    corners_src = np.float32([[0,0],[w_src,0],[w_src,h_src],[0,h_src]]).reshape(-1,1,2)
    corners_t   = cv2.perspectiveTransform(corners_src, H)
    corners_ref = np.float32([[0,0],[w_ref,0],[w_ref,h_ref],[0,h_ref]]).reshape(-1,1,2)

    all_corners = np.concatenate([corners_t, corners_ref], axis=0)
    xmin, ymin  = np.int32(all_corners.min(axis=0).ravel()) - 1
    xmax, ymax  = np.int32(all_corners.max(axis=0).ravel()) + 1

    # Décalage pour coordonnées positives
    shift = np.array([[1, 0, -xmin],
                      [0, 1, -ymin],
                      [0, 0,     1]], dtype=np.float64)

    canvas_w, canvas_h = xmax - xmin, ymax - ymin
    warped  = cv2.warpPerspective(img_src, shift @ H, (canvas_w, canvas_h))

    # Placer img_ref sur le canvas
    canvas = warped.copy()
    x0, y0 = -xmin, -ymin
    roi      = canvas[y0:y0+h_ref, x0:x0+w_ref]
    gray_roi = cv2.cvtColor(warped[y0:y0+h_ref, x0:x0+w_ref], cv2.COLOR_BGR2GRAY)
    mask_empty = gray_roi == 0                    # zones non couvertes par le warp

    roi[ mask_empty] = img_ref[ mask_empty]
    roi[~mask_empty] = cv2.addWeighted(
        roi[~mask_empty], 0.5, img_ref[~mask_empty], 0.5, 0)
    canvas[y0:y0+h_ref, x0:x0+w_ref] = roi
    return canvas


# ─────────────────────────────────────────────────────────────
# PANORAMA 6 IMAGES  
# ─────────────────────────────────────────────────────────────

def stitch_panorama(images, H_func):
    """
    Assemble N images en panorama en chaînant les homographies.

    Stratégie
    ---------
    - Image de référence : celle du milieu (index N//2)
    - Toutes les H_i→i+1 sont calculées avec RANSAC
    - On compose les H pour tout ramener dans le plan de l'image du milieu :
        H_0→ref = H_(ref-1)→ref · ... · H_0→1
        H_5→ref = inv(H_ref→5) = inv(H_4→5 · ... · H_ref→ref+1)
    - On warp chaque image sur un grand canvas commun
    """
    N   = len(images)
    ref = N // 2
    print(f"\n[panorama] {N} images, référence = img{ref}")

    # ── 1. Calculate all the matches between H_i→i+1 ──────────────────────
    H_pairs = {}
    for i in range(N - 1):
        print(f"\n  Matching img{i} ↔ img{i+1}")
        pts_i, pts_j, *_ = detect_and_match(images[i], images[i+1])
        H, mask = ransac_filter(H_func, pts_i, pts_j)
        err = compute_reprojection_error(H, pts_i[mask], pts_j[mask])
        print(f"  Erreur reprojection : {err:.2f} px")
        H_pairs[i] = H     

    # ── 2. Composition vers le plan de référence ──────────────
    # H_abs[i] = homographie qui envoie img_i → plan de ref
    H_abs = {ref: np.eye(3)}

    # Images à gauche de ref : H = H_(ref-1)→ref · ... · H_i→i+1
    for i in range(ref - 1, -1, -1):
        H_abs[i] = H_abs[i + 1] @ H_pairs[i]

    # Images à droite de ref : H = inv(H_ref→i)
    for i in range(ref + 1, N):
        H_abs[i] = np.linalg.inv(H_pairs[i - 1]) @ H_abs[i - 1]

    # ── 3. Calculate global canvas dimensions ────────────────────────────
    all_corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        all_corners.append(cv2.perspectiveTransform(corners, H_abs[i]))
    all_corners = np.concatenate(all_corners, axis=0)

    xmin, ymin = np.int32(all_corners.min(axis=0).ravel()) - 1
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel()) + 1

    shift = np.array([[1, 0, -xmin],
                      [0, 1, -ymin],
                      [0, 0,     1]], dtype=np.float64)

    canvas_w, canvas_h = xmax - xmin, ymax - ymin
    print(f"\n  Canvas : {canvas_w} × {canvas_h} px")

    # ── 4. Warp + blend net───────────────
    COLORS = [
        (0,   255, 255),   # jaune
        (0,   255,   0),   # vert
        (255,   0, 255),   # magenta
        (0,   128, 255),   # orange
        (255,   0,   0),   # bleu
        (0,     0, 255),   # rouge
    ]
 
    canvas  = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    overlay = np.zeros_like(canvas)              # canvas pour les contours
 
    for i, img in enumerate(images):
        H_final = shift @ H_abs[i]
        warped  = cv2.warpPerspective(img, H_final, (canvas_w, canvas_h))
 
        # Masque binaire des pixels valides
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask_w = gray > 0
 
        # Copie directe — écrase ce qui était là avant (pas de blend)
        canvas[mask_w] = warped[mask_w]
 
        # Contour de cette image
        # bin_mask = mask_w.astype(np.uint8) * 255
        # contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
        #                                cv2.CHAIN_APPROX_SIMPLE)
        # color = COLORS[i % len(COLORS)]
        # cv2.drawContours(overlay, contours, -1, color, 4)
 
    # Superposer les contours sur le canvas final
    contour_pixels = np.any(overlay > 0, axis=2)
    canvas[contour_pixels] = overlay[contour_pixels]
    return canvas

def stitch_panorama_homography(images, max_dim=8000):
    """
    Panorama N images avec cv2.findHomography + blend transparent.
 
    Problème fréquent : la composition de plusieurs H accumule des erreurs
    numériques → coins projetés à des coordonnées absurdes → canvas gigantesque.
 
    Solutions appliquées ici
    ------------------------
    1. Normalisation de chaque H (division par H[2,2])
    2. Clamp du canvas à max_dim × max_dim (défaut 8000 px)
       → les images hors canvas sont ignorées, pas de crash mémoire
    3. Affichage des coins projetés pour diagnostiquer la dérive
    """
    N   = len(images)
    ref = N // 2
    print(f"\n[panorama] {N} images, référence = img{ref}")
 
    # ── H_i → i+1 ────────────────────────────────────────────
    H_pairs = {}
    for i in range(N - 1):
        print(f"\n  Matching img{i} ↔ img{i+1}")
        pts_i, pts_j, *_ = detect_and_match(images[i], images[i+1])
        H, inlier_mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError(f"findHomography échoué pour img{i}↔img{i+1}")
        H = _normalize_H(H, pts_i)
        n_in = int(inlier_mask.sum()) if inlier_mask is not None else '?'
        print(f"  findHomography → {n_in}/{len(pts_i)} inliers")
        H_pairs[i] = H
 
    # ── Composition vers le plan de référence ─────────────────
    H_abs = {ref: np.eye(3, dtype=np.float64)}
    for i in range(ref - 1, -1, -1):
        H = H_abs[i+1] @ H_pairs[i]
        H_abs[i] = _normalize_H(H, None)
    for i in range(ref + 1, N):
        H = np.linalg.inv(H_pairs[i-1]) @ H_abs[i-1]
        H_abs[i] = _normalize_H(H, None)
 
    # ── Canvas global avec clamp ──────────────────────────────
    all_corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        proj = cv2.perspectiveTransform(c, H_abs[i])
        print(f"  img{i} coins projetés : {proj.reshape(4,2).astype(int).tolist()}")
        all_corners.append(proj)
 
    all_corners = np.concatenate(all_corners, axis=0)
    xmin, ymin  = np.int32(all_corners.min(axis=0).ravel()) - 1
    xmax, ymax  = np.int32(all_corners.max(axis=0).ravel()) + 1
 
    canvas_w = int(xmax - xmin)
    canvas_h = int(ymax - ymin)
    print(f"\n  Canvas brut : {canvas_w} × {canvas_h} px")
 
    # Clamp pour éviter l'erreur mémoire
    if canvas_w > max_dim or canvas_h > max_dim:
        print(f"  ⚠ Canvas trop grand — clamp à {max_dim} px (max_dim)")
        print(f"    Conseil : vérifiez que vos images se chevauchent bien")
        print(f"    et que les coins projetés ci-dessus ont l'air raisonnables.")
        scale  = max_dim / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)
        # Appliquer le même scale aux homographies
        S = np.diag([scale, scale, 1.0])
        for i in H_abs:
            H_abs[i] = S @ H_abs[i]
        xmin = int(xmin * scale)
        ymin = int(ymin * scale)
        print(f"  Canvas après clamp : {canvas_w} × {canvas_h} px")
 
    shift = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float64)
    print(f"  Canvas final : {canvas_w} × {canvas_h} px")
 
    # ── Blend transparent via accumulateur ────────────────────
    canvas_acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_acc = np.zeros((canvas_h, canvas_w),    dtype=np.float32)
 
    for i, img in enumerate(images):
        warped = cv2.warpPerspective(img, shift @ H_abs[i], (canvas_w, canvas_h))
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask_w = (gray > 0).astype(np.float32)
        canvas_acc += warped.astype(np.float32) * mask_w[:, :, np.newaxis]
        weight_acc += mask_w
 
    valid  = weight_acc > 0
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    result[valid] = (canvas_acc[valid] / weight_acc[valid, np.newaxis]).astype(np.uint8)
    return result
 

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main_panorama():
    images = load_images("stitch_pic/*.jpg")
    os.makedirs("output", exist_ok=True)

    # ── Panorama 6 images  ──
    print("\n══ PANORAMA 6 IMAGES (homography) ══")
    panorama = stitch_panorama_homography(images[2:5])
    show("Panorama 6 images", panorama)
    cv2.imwrite("output/panorama_5.jpg", panorama)
    print("Sauvegardé → output/panorama.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main_stitch2():
    images = load_images("stitch_pic/*.jpg")
    os.makedirs("output", exist_ok=True)

    img1, img2 = images[1], images[2]

    # ── Test sur 2 images ─────────────────────────────────────
    print("\n══ TEST 2 IMAGES ══")
    pts1, pts2, kp1, kp2, good = detect_and_match(img1, img2)

    for name, fn in [("translation", translation),
                     ("similarity",  similarity),
                     ("affine",      affine),
                     ("homography",  homography)]:
        print(f"\n── {name} ──")
        H, mask = ransac_filter(fn, pts1, pts2)
        err = compute_reprojection_error(H, pts1[mask], pts2[mask])
        print(f"  Erreur reprojection : {err:.2f} px")

        # Dessiner les matches (vert=inlier, rouge=outlier)
        # draw_matches(img1, img2, kp1, kp2, good,
        #              mask=mask,
        #              title=f"Matches — {name}",
        #              save_path=f"output/matches_{name}.jpg")

        # Stitch
        result = stitch_two(img1, img2, H)
        show(f"Stitch 2 images — {name}", result)
        cv2.imwrite(f"output/stitch12_{name}.jpg", result)
        print(f"  Sauvegardé → output/stitch12_{name}.jpg")

    print("\nPress key to stop")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main_mosaic():
    images = load_images("stitch_pic_low/*.jpg")
    mosaic_maker(images, name='mosaic_pic_low.png')

def main_mosaic_titles():
    images = load_images("output/stitch2_12/*.jpg")
    titles = ["affine", "homography", "similarity", "translation"]
    mosaic_maker_titles(images, titles=titles, name='mosaic_titles_2.png')
if __name__ == "__main__":
    main_mosaic_titles()
