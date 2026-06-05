
from module.display_helper import show, load_images, mosaic_maker, mosaic_maker_titles
from module.transformation import translation, similarity, affine, homography, compute_reprojection_error
from module.match import ransac_filter, detect_and_match
from module.track import build_track
from module.panorama_stitch import stitch_panorama_homography, stitch_two
import os
import cv2
import numpy as np

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

def main_tracks():
    images = load_images("stitch_pic/*.jpg")
    tracks = build_track(images)
    print(f"{len(tracks)} tracks construits à partir de {len(images)} images")

if __name__ == "__main__":
    main_tracks()
