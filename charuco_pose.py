"""
Framework   : OpenCV Aruco
Description : Calibration of camera and using that for finding pose of multiple markers
Status      : Working
References  :
    1) https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
    2) https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import glob
from tqdm import tqdm
from print_charuco import CHARUCO_BOARD, CHARUCO_DICT, CHARUCO_PARAMS


def get_boundaries(points, y0):
    distances1 = np.sqrt(
        points[:, :, 0] * points[:, :, 0] + points[:, :, 1] * points[:, :, 1]
    )
    boundary = np.zeros((4, 1, 2))
    boundary[0, :, :] = points[np.argmin(distances1)]
    boundary[3, :, :] = points[np.argmax(distances1)]

    d_y = y0 - points[:, :, 1]
    distances1 = np.sqrt(points[:, :, 0] * points[:, :, 0] + d_y * d_y)
    boundary[2, :, :] = points[np.argmin(distances1)]
    boundary[1, :, :] = points[np.argmax(distances1)]
    return boundary


def get_box(boundaries):
    box = np.zeros((4, 1, 2))
    w = boundaries[1, 0, 0] - boundaries[0, 0, 0]
    box[0] = boundaries[0]
    box[1] = boundaries[0] + np.array([[w, 0]])
    box[2] = boundaries[0] + np.array([[0, w]])
    box[3] = boundaries[0] + np.array([[w, w]])
    return box


def draw_point(image, coord):
    return cv2.circle(image, coord, radius=5, color=(0, 0, 255), thickness=-1)


####---------------------- CALIBRATION ---------------------------
# termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# checkerboard of size (7 x 6) is used
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# iterating through all calibration images
# in the folder
images = glob.glob("calib_images/checkerboard/*.jpg")

print("Callibrating camera...")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the chess board (calibration pattern) corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # if calibration pattern is found, add object points,
    # image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refine the corners of the detected corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)


ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

rvec = rvecs[0]
tvec = tvecs[0]
print("Done.")


###------------------ ARUCO TRACKER ---------------------------
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
print("Starting loop...")

pbar = tqdm()
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    print(frame.shape)
    # operations on the frame

    if ret == True:

        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )  # aruco.detectMarkers() requires gray image

        CHARUCO_PARAMS.adaptiveThreshConstant = 10
        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, CHARUCO_DICT, parameters=CHARUCO_PARAMS
        )

        aruco.refineDetectedMarkers(
            gray, CHARUCO_BOARD, corners, ids, rejectedImgPoints
        )

        if np.all(ids != None):  # if there is at least one marker detected
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(
                corners, ids, gray, CHARUCO_BOARD
            )

            # try to create a birds eye view of the scene based on the charuco recognition
            try:

                main_corners = get_boundaries(charucoCorners, frame.shape[0])
                # pbar.set_description(f"corners: {main_corners}")
                for item in main_corners:
                    draw_point(frame, (int(item[0, 0]), int(item[0, 1])))
                print(f"corners: {main_corners}")

                new_corners = get_box(main_corners)
                print(f"box: ", get_box(new_corners))

                transform_matrix, _ = cv2.findHomography(
                    main_corners.astype(np.int), new_corners.astype(np.int)
                )

                # transform_matrix = cv2.getPerspectiveTransform(
                #     main_corners, new_corners
                # )
                w, h, _ = frame.shape
                warped = cv2.warpPerspective(frame, transform_matrix, (h, w))
                cv2.imshow("warped", warped)
            except Exception as e:
                print(e)
            im_with_charuco_board = aruco.drawDetectedCornersCharuco(
                frame, charucoCorners, charucoIds, (0, 255, 0)
            )
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(
                charucoCorners,
                charucoIds,
                CHARUCO_BOARD,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
            )  # posture estimation from a charuco board
            if retval == True:
                im_with_charuco_board = aruco.drawAxis(
                    im_with_charuco_board, camera_matrix, dist_coeffs, rvec, tvec, 100
                )  # axis length 100 can be changed according to your requirement
        else:
            im_with_charuco_board = frame

        cv2.imshow("charucoboard", im_with_charuco_board)

        if cv2.waitKey(2) & 0xFF == ord("q"):
            break
    else:
        break

    pbar.update(1)


cap.release()  # When everything done, release the capture
cv2.destroyAllWindows()
