import cv2
import numpy as np
import glob 
from ex4.module.transformation import translation, similarity, affine


# Qestion 1
images = sorted(glob.glob('stitch_pic/*.jpg'))

#Question 2
# Initialize SIFT detector
sift = cv2.SIFT_create()
kp_list = []
des_list = []

for i in range(len(images)):
    img = cv2.imread(images[i])
    kp, des = sift.detectAndCompute(img, None)
    kp_list.append(kp)
    des_list.append(des)


#Using this matcher and not BFMatcher because it is faster for large datasets
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

matches_dict = {}
for i in range(len(des_list) - 1):
    # Match image i with image i+1 using Lowe's Ratio Test (knnMatch)
    raw_matches = matcher.knnMatch(des_list[i], des_list[i+1], k=2)
    
    # Filter good matches
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
            
    # Store the result dynamically
    matches_dict[f"match_{i}_to_{i+1}"] = good_matches


# Question 3 
pts_pair = {}
for i in range(len(images) - 1):
    matches = matches_dict[f"match_{i}_to_{i+1}"]
    
    pts_current = np.float32([kp_list[i][m.queryIdx].pt for m in matches])
    pts_next    = np.float32([kp_list[i+1][m.trainIdx].pt for m in matches])
    
    # Sauvegarde pour l'Étape 4
    pts_pair[f"pts_{i}_to_{i+1}"] = (pts_current, pts_next)


# Question 5
def show_image(title, img, max_width=1280):
    scale = min(1.0, max_width / img.shape[1])
    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imshow(title, small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#H, mask = cv2.findHomography(pts_pair["pts_0_to_1"][1], pts_pair["pts_0_to_1"][0], cv2.RANSAC, 5.0)
#H = translation(pts_pair["pts_0_to_1"][1], pts_pair["pts_0_to_1"][0])
#H = similarity(pts_pair["pts_0_to_1"][0], pts_pair["pts_0_to_1"][1])
# print(H)

# H_inv= np.linalg.inv(H)
# # Canvas adapté au décalage réel
# dx = int(H_inv[0, 2])
# dy = int(H_inv[1, 2])
# canvas_w = w1 + abs(dx)
# canvas_h = max(h1, h2 + abs(dy))

# #img1_warped = cv2.warpPerspective(img1, H, (w1 + w2, h1))
# img1_warped = cv2.warpPerspective(img1, H_inv, (canvas_w, canvas_h))
# result = img1_warped.copy()
# result[0:h1, 0:w1] = img0

# img0 = cv2.imread(images[0])
# img1 = cv2.imread(images[1])
# h0, w0 = img0.shape[:2]
# h1, w1 = img1.shape[:2]

# # H : pts0 -> pts1, donc H_inv place img1 dans le repère de img0
# H = similarity(pts_pair["pts_0_to_1"][0], pts_pair["pts_0_to_1"][1])
# H_inv = np.linalg.inv(H)

img0 = cv2.imread(images[0])
img1 = cv2.imread(images[1])

img_matches = cv2.drawMatches(
    img0, kp_list[0],
    img1, kp_list[1],
    matches_dict["match_0_to_1"][:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
# cv2.imwrite("matches.jpg", img_matches)
# print("saved to matches.jpg")

pts0, pts1 = pts_pair["pts_0_to_1"]