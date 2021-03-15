import cv2
import cv2.aruco as aruco

# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 4
CHARUCOBOARD_COLCOUNT = 4
# CHARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)
# CHARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_50)
CHARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)
# ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_7X7_1000)
# CHARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

# Create constants to be passed into OpenCV and Aruco methods
# Length of the squares and markers must be in the same units.
CHARUCO_BOARD = aruco.CharucoBoard_create(
    squaresX=CHARUCOBOARD_COLCOUNT,
    squaresY=CHARUCOBOARD_ROWCOUNT,
    # squareLength=0.19134,
    # markerLength=0.1424,
    # squareLength=0.9,
    # markerLength=0.8,
    # squareLength=40,
    # markerLength=30,
    squareLength=90,
    markerLength=75,
    dictionary=CHARUCO_DICT,
)
CHARUCO_PARAMS = aruco.DetectorParameters_create()


# board = aruco.CharucoBoard_create(5, 7, squareLength, markerLength, aruco_dict)


if __name__ == "__main__":
    # print(CHARUCO_BOARD.dictionary)
    charuco_board_image = CHARUCO_BOARD.draw((600, 500), marginSize=10, borderBits=1)
    charuco_path = f"images/charuco_board_{CHARUCOBOARD_ROWCOUNT}x{CHARUCOBOARD_COLCOUNT}_{aruco.DICT_4X4_50}.png"
    cv2.imwrite(
        charuco_path,
        charuco_board_image,
    )
    print(f"{charuco_path}, Done!")