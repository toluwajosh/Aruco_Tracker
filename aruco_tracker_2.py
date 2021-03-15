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
from print_charuco import CHARUCO_BOARD


####---------------------- CALIBRATION ---------------------------
# load calibration parameters
cv_file = cv2.FileStorage("calib_images/logicool_hd_1080p.yaml", cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()
cv_file.release()
print("Finished calibration..")

###------------------ ARUCO TRACKER ---------------------------
cap = cv2.VideoCapture(1)
pbar = tqdm()
print("Starting loop...")
while True:
    ret, frame = cap.read()

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    aruco_dict = CHARUCO_BOARD.dictionary

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        # retval, rvec, tvec = aruco.estimatePoseBoard(
        #     corners, ids, CHARUCO_BOARD, mtx, dist
        # )
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(
            charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec
        )
        # (rvec-tvec).any() # get rid of that nasty numpy value array error

        for i in range(0, ids.size):
            # draw axis for the aruco markers
            aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)

        # code to show ids of the marker found
        strg = ""
        for i in range(0, ids.size):
            strg += str(ids[i][0]) + ", "

        cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow("frame", frame)
    pbar.update(1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
