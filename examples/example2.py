# THE IMPLEMENTING OF THE COMPUTER VISION PART IN THE RTDE PROGRAM
# the position x and y are sent successfully
# the distance between the fingers control the gripper

import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import cv2
from examples.CamSetup import setup_camera
import matplotlib.pyplot as plt
import pickle
import numpy as np

sys.path.append("..")

logging.basicConfig(level=logging.DEBUG)

ROBOT_HOST = "192.168.0.3"
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"

logging.getLogger().setLevel(logging.INFO)

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


def setp_to_list(sp):
    sp_list = []
    for i in range(0, 6):
        sp_list.append(sp.__dict__["input_double_register_%i" % i])
    return sp_list


def list_to_setp(sp, list):
    for i in range(0, 6):
        sp.__dict__["input_double_register_%i" % i] = list[i]
    return sp


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

def valable(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax):
    if (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax):
        return True
    else :
        return False
def setup_robot():
    conf = rtde_config.ConfigFile(config_filename)
    state_names, state_types = conf.get_recipe("state")
    setp_names, setp_types = conf.get_recipe("setp")

    watchdog_names, watchdog_types = conf.get_recipe("watchdog")

    con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
    con.connect()

    test = con.is_connected()
    print("is_connected: ", test)
    con.send_start()
    if test:
        # if connection_state:
        print('Connected to the robot')
    else:
        print('Failed to connect to the robot')
        sys.exit()

    # get controller version
    con.get_controller_version()
    print('setp_names', setp_names)
    print('setp types', setp_types)

    # setup recipes
    con.send_output_setup(state_names, state_types)
    setp = con.send_input_setup(setp_names, setp_types)
    watchdog = con.send_input_setup(watchdog_names, watchdog_types)

    return con, setp, watchdog


def main():
    con, setp, watchdog = setup_robot()
    if not con.send_start():
        sys.exit()

    device, qRgb = setup_camera()
    aruco_type = "DICT_5X5_100"
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

    cv2.namedWindow('Estimated Pose')


    with open('calibration.pkl', 'rb') as f:
        matrix_coefficients, distortion_coefficients = pickle.load(f)

    plt.ion()  # Enable interactive mode
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 0.5])

    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Position')
    scatter = ax.scatter([], [], [], c='r', marker='o')

    tvecs_list = []
    rvecs_list = []
    robot_x_max = 0.75
    robot_x_min = 0.4
    robot_y_max = 0.21
    robot_y_min = -0.21
    robot_z_max = 0.3
    robot_z_min = 0.003


    setp.input_double_register_0 = 0
    setp.input_double_register_1 = 0
    setp.input_double_register_2 = 0
    setp.input_double_register_3 = 0
    setp.input_double_register_4 = 0
    setp.input_double_register_5 = 0
    setp.input_double_register_6 = 0

    watchdog.input_int_register_0 = 0


    while True:
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()

        # Process the frame for hand detection

        h, w, c = frame.shape
        width = 1000
        height = int(width * (h / w))
        #frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

        state = con.receive()
        if state is None:
            print("Failed to receive state")
            break

        corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=cv2.aruco.DetectorParameters())

        detected_markers = aruco_display(corners, ids, rejected, frame)

        output, rvec, tvec = pose_estimation(frame, ARUCO_DICT[aruco_type], matrix_coefficients,
                                             distortion_coefficients)
        # print('rvec = ', rvec, 'tvec = ', tvec)
        if rvec is not None and len(rvec) > 0:
            rvecs_list.append(rvec[0][0])
        if tvec is not None and len(tvec) > 0:
            tvecs_list.append(tvec[0][0])

        if rvecs_list and tvecs_list:
            rvecs_array = np.array(rvecs_list)
            tvecs_array = np.array(tvecs_list)

            x = scale(tvecs_array[:, 0], 0.16, -0.011, robot_x_min, robot_x_max)
            y = scale(tvecs_array[:, 1], -0.14, -0.029, robot_y_min, robot_y_max)
            z = scale(tvecs_array[:, 2], 3.06, 1.8, robot_z_min, robot_z_max)
            Rx = rvecs_array[:, 0]
            Ry = rvecs_array[:, 1]
            Rz = rvecs_array[:, 2]
            data = [float(x[-1]), float(y[-1]), float(z[-1]), 0, 3.11, 0.04]
            #data = [float(x[-1]), float(y[-1]), float(z[-1]),float(Rx[-1]), float(Ry[-1]), float(Rz[-1])]
            #scatter._offsets3d = (Rx, Ry, Rz)
            # scatter._offsets3d = (x, y, z)
            fig.canvas.draw()
            fig.canvas.flush_events()
            print(data)



            if valable(x[-1],y[-1],z[-1],robot_x_min,robot_x_max,robot_y_min,robot_y_max,robot_z_min,robot_z_max):
                list_to_setp(setp, data)
                con.send(setp)

                con.send(watchdog)

        cv2.imshow('Estimated Pose', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    plt.show()
    cv2.destroyAllWindows()
    plt.ioff()
    con.send_pause()
    con.disconnect()
    device.close()


if __name__ == '__main__':
    main()
