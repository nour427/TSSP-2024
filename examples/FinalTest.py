import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import cv2
from examples.CamSetup import setup_camera
import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
import mediapipe as mp
from collections import deque

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
    for i in range(0, 7):
        sp.__dict__["input_double_register_%i" % i] = list[i]
    return sp


def aruco_display(corners, ids, rejected, image):
    detected = False
    if len(corners) > 0:
        detected = True
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
            #print("[Inference] ArUco marker ID: {}".format(markerID))

    return image, detected


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
    else:
        return


class HandDetector():

    def __init__(self, detectionCon=0.5, trackCon=0.5):

        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


    def getIndexFingerMCPPosition(self):
        if self.results and self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                lm = handLMS.landmark[5]
                return lm.x, lm.y, lm.z
        return None, None, None

    def getdistbetweenfingers(self, img):
        dist = 0

        if self.results and self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLMS.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 4:  # this is the thumbq
                        # lets draw a point
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), -1)
                        thumbX, thumbY = cx, cy

                    if id == 8:  # this is the finger tip
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), -1)
                        fingerTipX, fingerTipY = cx, cy

                dist = math.sqrt(((thumbX - fingerTipX) ** 2) + ((thumbY - fingerTipY) ** 2))

                cv2.line(img, (thumbX, thumbY), (fingerTipX, fingerTipY), (0, 0, 255), 2)

        return img, dist

    def getOrientation(self):
        if self.results and self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                lm0 = handLMS.landmark[0]
                lm5 = handLMS.landmark[5]

                # Calculate vector from landmark 0 to landmark 5
                vector_x = lm5.x - lm0.x
                vector_y = lm5.y - lm0.y

                angle = math.atan2(vector_y, vector_x)

                return angle
        return None


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
    detector = HandDetector()
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

    robot_x_max = 0.75
    robot_x_min = 0.4
    robot_y_max = 0.21
    robot_y_min = -0.21
    robot_z_max = 0.3
    robot_z_min = 0.003

    buffer_size = 10
    x_buffer = deque(maxlen=buffer_size)
    y_buffer = deque(maxlen=buffer_size)
    z_buffer = deque(maxlen=buffer_size)
    orientation_buffer = deque(maxlen=buffer_size)
    dist_buffer = deque(maxlen=buffer_size)

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
        frame = detector.findHands(frame)
        frame, dist = detector.getdistbetweenfingers(frame)

        h, w, c = frame.shape
        width = 1000
        height = int(width * (h / w))
        #frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

        state = con.receive()
        if state is None:
            print("Failed to receive state")
            break

        x, y, z = detector.getIndexFingerMCPPosition()
        angle = detector.getOrientation()

        corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=cv2.aruco.DetectorParameters())

        detected_markers, detected = aruco_display(corners, ids, rejected, frame)

        output, rvec, tvec = pose_estimation(frame, ARUCO_DICT[aruco_type], matrix_coefficients,
                                             distortion_coefficients)

        if tvec is not None and len(tvec) > 0:
            tvecs_list.append(tvec[0][0])

        if tvecs_list and x and y and z and detected and dist and angle is not None:
            tvecs_array = np.array(tvecs_list)

            tx = scale(tvecs_array[:, 0], 0.16, -0.011, robot_x_min, robot_x_max)
            ty = scale(tvecs_array[:, 1], -0.14, -0.029, robot_y_min, robot_y_max)
            tz = scale(tvecs_array[:, 2], 3.06, 1.8, robot_z_min, robot_z_max)

            x = scale(x, 0.7, 0.24, robot_x_min, robot_x_max)
            y = scale(y, 0.14, 0.77, robot_y_min, robot_y_max)

            x_buffer.append(x)
            y_buffer.append(y)
            z_buffer.append(float(tz[-1]))

            avg_x = np.mean(x_buffer, axis=0)
            avg_y = np.mean(y_buffer, axis=0)
            avg_z = np.mean(z_buffer, axis=0)

            dist_buffer.append(dist)
            avg_dist = np.mean(dist_buffer, axis=0)

            orientation_buffer.append(angle)
            avg_angle = np.mean(orientation_buffer, axis=0)

            scatter._offsets3d = (tx, ty, tz)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # data = [x, y, float(tz[-1]), 0, 3.11, 0.04, dist]
            data = [x, y, float(tz[-1]), - angle, 0, 0, dist]
            avg_data = [avg_x, avg_y, avg_z, -avg_angle, 0, 0, avg_dist]
            cv2.putText(frame, f"Data: {data}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Data: {avg_data}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if valable(avg_x, avg_y, avg_z, robot_x_min, robot_x_max, robot_y_min, robot_y_max, robot_z_min,
                       robot_z_max):
                print('data', avg_data)
                list_to_setp(setp, avg_data)
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
