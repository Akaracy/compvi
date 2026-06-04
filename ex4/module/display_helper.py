import math
import numpy as np
import cv2 as cv2
import glob


# ─────────────────────────────────────────────────────────────
# UTILITAIRE AFFICHAGE
# ─────────────────────────────────────────────────────────────

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
