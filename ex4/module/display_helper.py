import math
import numpy as np
import cv2 as cv2
import glob


# ───────────────────────────────────────────────────────────── 
# UTILITAIRE AFFICHAGE
# ─────────────────────────────────────────────────────────────

def load_images(pattern="stitch_pic/*.jpg"):
    paths = sorted(glob.glob(pattern))
    if len(paths) < 2:
        raise FileNotFoundError(f"Besoin d'au moins 2 images : '{pattern}'")
    imgs = [cv2.imread(p) for p in paths]
    print(f"[load] {len(imgs)} images chargées")
    return imgs

def show(title, img, max_w=1300, max_h=700):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)


# ------------------------------------------------------ 
# Utilitaire pour créer des images de rapport
# ------------------------------------------------------

def mosaic_maker(images, name='mosaic.png'):

    if images:
        n = len(images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        # Resize all images to the same size (use the first image's dimensions)
        h, w = images[0].shape[:2]
        thumb_w, thumb_h = w // 2, h // 2  # scale down to keep mosaic manageable

        resized = [cv2.resize(img, (thumb_w, thumb_h)) for img in images]

        # Pad with black images if needed to fill the grid
        blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
        while len(resized) < rows * cols:
            resized.append(blank)

        # Assemble rows then stack vertically
        row_imgs = []
        for r in range(rows):
            row_imgs.append(np.hstack(resized[r * cols:(r + 1) * cols]))
        mosaic = np.vstack(row_imgs)

        cv2.imwrite(name, mosaic)
        print(f"Mosaic saved as {name} ({cols}x{rows} grid, {n} images)")


def mosaic_maker_titles(images, titles, name="mosaic.png",
                 font_scale=1.2,
                 text_height=100):

    if not images:
        return

    n = len(images)

    if len(titles) != n:
        raise ValueError(
            f"{len(titles)} titres fournis pour {n} images."
        )

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Taille des miniatures
    h, w = images[0].shape[:2]
    thumb_w, thumb_h = w // 2, h // 2

    resized = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3

    for img, title in zip(images, titles):

        img = cv2.resize(img, (thumb_w, thumb_h))

        # Overlay semi-transparent
        overlay = img.copy()

        cv2.rectangle(
            overlay,
            (0, thumb_h - text_height),
            (thumb_w, thumb_h),
            (0, 0, 0),
            -1
        )

        alpha = 0.6
        img = cv2.addWeighted(
            overlay, alpha,
            img, 1 - alpha,
            0
        )

        # Centrage du texte
        text_size, _ = cv2.getTextSize(
            title,
            font,
            font_scale,
            thickness
        )
        text_height = max(50, text_size[1] + 20)
        text_x = max((thumb_w - text_size[0]) // 2, 5)
        text_y = thumb_h - (text_height - text_size[1]) // 2

        cv2.putText(
            img,
            title,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        resized.append(img)

    # Case vide pour compléter la grille
    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)

    while len(resized) < rows * cols:
        resized.append(blank)

    # Construction de la mosaïque
    row_imgs = []
    for r in range(rows):
        row_imgs.append(
            np.hstack(
                resized[r * cols:(r + 1) * cols]
            )
        )

    mosaic = np.vstack(row_imgs)

    cv2.imwrite(name, mosaic)

    print(
        f"Mosaic saved as {name} "
        f"({cols}x{rows} grid, {n} images)"
    )
