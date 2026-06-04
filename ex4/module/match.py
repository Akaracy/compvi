import numpy as np
import cv2
import glob

# ─────────────────────────────────────────────────────────────
# DETECTION + MATCHING
# ─────────────────────────────────────────────────────────────

def detect_and_match(img1, img2, ratio=0.75):
    """SIFT + FLANN avec test de ratio de Lowe. Retourne (pts1, pts2, kp1, kp2, good_matches)."""
    sift = cv2.SIFT_create() #matcher
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )
    raw = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance] #lowe ratio

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    print(f"  {len(good)} bons matches")
    return pts1, pts2, kp1, kp2, good


# ─────────────────────────────────────────────────────────────
# DRAW MATCHES
# ─────────────────────────────────────────────────────────────

def draw_matches(img1, img2, kp1, kp2, good_matches, mask=None,
                 max_draw=50, title="Matches", save_path=None):
    """
    Affiche les correspondances entre img1 et img2.

    Paramètres
    ----------
    mask      : tableau bool (len = len(good_matches)) — True = inlier (vert), False = outlier (rouge)
                Si None, tous les matches sont dessinés en vert.
    max_draw  : limite le nombre de lignes pour la lisibilité
    """
    # Couleurs : inliers vert, outliers rouge
    if mask is not None:
        draw_params_in  = dict(matchColor=(0, 255, 0),
                               singlePointColor=None,
                               matchesMask=mask.astype(int).tolist(),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        draw_params_out = dict(matchColor=(0, 0, 255),
                               singlePointColor=None,
                               matchesMask=(~mask).astype(int).tolist(),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        vis = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params_in)
        vis = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, vis,  **draw_params_out)
    else:
        # Sous-ensemble aléatoire pour ne pas surcharger
        subset = good_matches[:max_draw]
        vis = cv2.drawMatches(img1, kp1, img2, kp2, subset, None,
                              matchColor=(0, 255, 0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Redimensionner pour l'affichage
    h, w = vis.shape[:2]
    scale = min(1400 / w, 600 / h, 1.0)
    if scale < 1.0:
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cv2.imshow(title, vis)
    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"  Matches sauvegardés → {save_path}")
    return vis


# ─────────────────────────────────────────────────────────────
# RANSAC
# ─────────────────────────────────────────────────────────────

# Nombre minimal de points par modèle pour estimer H
_MIN_PTS = {"translation": 1, "similarity": 2, "affine": 3}

def ransac_filter(H_func, pts_src, pts_dst, threshold=4.0, iterations=1000):
    """
    RANSAC générique pour nos 3 transformations.

    Principe
    --------
    À chaque itération :
      1. Tire aléatoirement le minimum de points nécessaires
      2. Estime H sur ce petit sous-ensemble
      3. Projette TOUS les points avec H
      4. Compte les inliers (erreur de reprojection < threshold)
    Garde le meilleur H, ré-estime avec tous les inliers.

    Retourne (H_final, mask_bool)
    """
    n_min = _MIN_PTS.get(H_func.__name__, 4)
    n = len(pts_src)
    best_H, best_mask = None, np.zeros(n, dtype=bool)

    for _ in range(iterations):
        idx = np.random.choice(n, n_min, replace=False)
        try:
            H = H_func(pts_src[idx], pts_dst[idx])
        except np.linalg.LinAlgError:
            continue

        # Reprojection de tous les points
        src_h = np.c_[pts_src, np.ones(n)].T          # 3 × n
        proj  = (H @ src_h).T                          # n × 3
        proj /= proj[:, [2]]
        errors = np.linalg.norm(proj[:, :2] - pts_dst, axis=1)
        mask = errors < threshold

        if mask.sum() > best_mask.sum():
            best_mask = mask
            best_H = H

    # Ré-estimation finale sur tous les inliers
    if best_mask.sum() >= n_min:
        best_H = H_func(pts_src[best_mask], pts_dst[best_mask])

    print(f"  RANSAC → {best_mask.sum()}/{n} inliers")
    return best_H, best_mask
