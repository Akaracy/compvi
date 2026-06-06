import numpy as np
import cv2 as cv
import glob
import math

def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

def draw_pyramid(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor 
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,0,255),-3)

    for i in zip(range(4)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[4]),(255),3)
    
    return img

def draw_shape(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor 
    img = cv.drawContours(img, [imgpts[:4]],-1,(255,0,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    
    return img

def draw_shape2(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor 
    img = cv.drawContours(img, [imgpts[:8]],-1,(255,0,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(8),range(8,16)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[8:]],-1,(0,0,255),3)
    
    return img

# Load previously saved data
with np.load('calibration_data.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

axis = np.float32([[3,0,0] , [0,3,0], [0,0,-3]]).reshape(-1,3) #blue (down), green (left), red
axis_4cube = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
axis_4pyramid= np.float32([[3,3,0], [5,3,0], [5,5,0], [3,5,0],
                   [4,4,-2] ])
axis_4shape= np.float32([[0.5,0.5,0], [0.5,2.5,0], [2.5,2.5,0], [2.5,0.5,0],
                   [0,3,-3],[3,3,-3],[3,0,-3], [0,0,-3] ])
axis_4shape2 = np.float32([[0.5, 1, 0], [1, 0.5, 0], [2, 0.5, 0], [2.5, 1, 0], [2.5, 2, 0], [2, 2.5, 0], [1, 2.5, 0], [0.5, 2,0],
                           [1, 0, -2.5], [2, 0, -2.5], [3, 1, -2.5], [3, 2, -2.5], [2, 3, -2.5], [1, 3, -2.5], [0, 2, -2.5], [0, 1, -2.5]])
axis_4shape3 = np.float32([[5.5, 1, 0], [6, 0.5, 0], [7, 0.5, 0], [7.5, 1, 0], [7.5, 2, 0], [7, 2.5, 0], [6, 2.5, 0], [5.5, 2,0],
                           [6, 0, -2.5], [7, 0, -2.5], [8, 1, -2.5], [8, 2, -2.5], [7, 3, -2.5], [6, 3, -2.5], [5, 2, -2.5], [5, 1, -2.5]])


processed_images = []

for fname in glob.glob('calibration_pic/*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # Project and draw
        # impts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # img = draw_axis(img, corners2, impts)

        impts_cube, jac = cv.projectPoints(axis_4cube, rvecs, tvecs, mtx, dist)
        img = draw_cube(img, corners2, impts_cube)

        imgpts_shape1, jac = cv.projectPoints(axis_4shape3, rvecs, tvecs, mtx, dist)
        img = draw_shape2(img, corners2, imgpts_shape1)

        imgpts_shape, jac = cv.projectPoints(axis_4shape, rvecs, tvecs, mtx, dist)
        img = draw_shape(img, corners2, imgpts_shape)

        imgpts_pyramid, jac = cv.projectPoints(axis_4pyramid, rvecs, tvecs, mtx, dist)
        img = draw_pyramid(img, corners2, imgpts_pyramid)

        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.imshow('img', img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:6]+'.png', img)
        processed_images.append(img)

# --- Build and save mosaic ---
# if processed_images:
#     n = len(processed_images)
#     cols = math.ceil(math.sqrt(n))
#     rows = math.ceil(n / cols)

#     # Resize all images to the same size (use the first image's dimensions)
#     h, w = processed_images[0].shape[:2]
#     thumb_w, thumb_h = w // 2, h // 2  # scale down to keep mosaic manageable

#     resized = [cv.resize(img, (thumb_w, thumb_h)) for img in processed_images]

#     # Pad with black images if needed to fill the grid
#     blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
#     while len(resized) < rows * cols:
#         resized.append(blank)

#     # Assemble rows then stack vertically
#     row_imgs = []
#     for r in range(rows):
#         row_imgs.append(np.hstack(resized[r * cols:(r + 1) * cols]))
#     mosaic = np.vstack(row_imgs)

#     cv.imwrite('mosaic_axis.png', mosaic)
#     print(f"Mosaic saved as mosaic.png ({cols}x{rows} grid, {n} images)")
cv.destroyAllWindows()