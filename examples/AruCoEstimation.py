import numpy as np
import cv2
from matplotlib import pyplot as plt
from examples.CamSetup import setup_camera
import pickle
from collections import deque
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:

        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))

    return image


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    rvec = []
    tvec = []
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if len(corners) > 0:
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                distortion_coefficients)

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            frame = cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame, rvec, tvec


def scale(pos, min, max, robotmin, robotmax):
    return robotmin + (pos - min) * (robotmax - robotmin) / (max - min)



def main():
    device, qRgb = setup_camera()

    aruco_type = "DICT_5X5_100"

    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    with open('calibration.pkl', 'rb') as f:
        matrix_coefficients, distortion_coefficients = pickle.load(f)

    plt.ion()  # Enable interactive mode
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-1, 1])

    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Position')
    scatter = ax.scatter([], [], [], c='r', marker='o')
    x_vals, y_vals, z_vals = [], [], []
    tvecs_list = []
    rvecs_list =[]
    robot_x_max = 0.75
    robot_x_min = 0.4
    robot_y_max = 0.21
    robot_y_min = -0.21
    robot_z_max = 0.3
    robot_z_min = 0.003
    rvecs_list = []
    buffer_size = 10
    z_buffer = deque(maxlen=buffer_size)
    while True:
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()

        h, w, c = frame.shape
        width = 1000
        height = int(width * (h / w))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=cv2.aruco.DetectorParameters())

        detected_markers = aruco_display(corners, ids, rejected, frame)

        output, rvec, tvec = pose_estimation(frame, ARUCO_DICT[aruco_type], matrix_coefficients,
                                             distortion_coefficients)
        #print('rvec = ', rvec, 'tvec = ', tvec)

        if tvec is not None and len(tvec) > 0:
            tvecs_list.append(tvec[0][0])

        if rvec is not None and len(rvec) > 0:
            rvecs_list.append(rvec[0][0])

        if rvecs_list:
            rvecs_array = np.array(rvecs_list)

            Rx = rvecs_array[:, 0]
            Ry = rvecs_array[:, 1]
            Rz = rvecs_array[:, 2]


            scatter._offsets3d = (Rx, Ry, Rz)
            fig.canvas.draw()
            fig.canvas.flush_events()
            print('ROTATION VECTOR ',rvecs_array)


        if tvecs_list:
            tvecs_array = np.array(tvecs_list)

            x = scale(tvecs_array[:, 0],  0.065,-0.1, robot_x_min, robot_x_max)
            y = scale(tvecs_array[:, 1], -0.29, -0.17, robot_y_min, robot_y_max)
            z = scale(tvecs_array[:, 2], 5.95, 3.4, robot_z_min, robot_z_max)
            #if x is not None and y is not None and z is not None:

              #  scaled_tvecs = [[float(x[i]),float(y[i]),float(z[i]),0, 3.11, 0.04] for i in range(len(x))]
               # print (scaled_tvecs)
            data =[float(x[-1]),float(y[-1]),float(z[-1]),0, 3.11, 0.04]
            z_buffer.append(float(z[-1]))
            avg_z = np.mean(z_buffer, axis=0)
            cv2.putText(frame, f"z: {float(z[-1])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"z avg : {avg_z}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


            #scatter._offsets3d = (x, y, z)
            #fig.canvas.draw()
            #fig.canvas.flush_events()

        cv2.imshow('Estimated Pose', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    plt.ioff()

    plt.show()
    cv2.destroyAllWindows()
    device.close()


if __name__ == '__main__':
    main()
