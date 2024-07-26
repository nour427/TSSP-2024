# THE IMPLEMENTING OF THE COMPUTER VISION PART IN THE RTDE PROGRAM
# the position x and y are sent successfully
# the distance between the fingers control the gripper

import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import cv2
import mediapipe as mp
from examples.CamSetup import setup_camera
import matplotlib.pyplot as plt
import math

sys.path.append("..")

logging.basicConfig(level=logging.DEBUG)

ROBOT_HOST = "192.168.0.3"
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"

logging.getLogger().setLevel(logging.INFO)


def setp_to_list(sp):
    sp_list = []
    for i in range(0, 6):
        sp_list.append(sp.__dict__["input_double_register_%i" % i])
    return sp_list


def list_to_setp(sp, list):
    for i in range(0, 7):
        sp.__dict__["input_double_register_%i" % i] = list[i]
    return sp


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

    def getHandPosition(self, img, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLMS.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)  # get the X,Y of a finger
                    lmlist.append([id, cx, cy])
                    if draw:
                        cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return lmlist

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


def valable(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax):
    if (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax):
        return True
    else :
        return

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
    # start data synchronization
    if not con.send_start():
        sys.exit()

    device, qRgb = setup_camera()
    detector = HandDetector()
    cv2.namedWindow('OAK-D RGB with Mediapipe')

    plt.ion()  # Enable interactive mode
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Index Finger MCP Coordinates')  # Set title for the plot
    scatter = ax.scatter([], [], [], c='r', marker='o')

    fig_dist, ax_dist = plt.subplots()
    ax_dist.set_xlabel('X time')
    ax_dist.set_ylabel('Y dist')
    ax_dist.set_title('distance')
    dist_vals = []

    x_vals, y_vals, z_vals = [], [], []
    bounds = (-1, 1, -1, 1, -1, 1)



    setp.input_double_register_0 = 0
    setp.input_double_register_1 = 0
    setp.input_double_register_2 = 0
    setp.input_double_register_3 = 0
    setp.input_double_register_4 = 0
    setp.input_double_register_5 = 0
    setp.input_double_register_6 = 0
    robot_x_max = 0.75
    robot_x_min = 0.4
    robot_y_max = 0.21
    robot_y_min = -0.21
    robot_z_max = 0.3
    robot_z_min = 0.003



    watchdog.input_int_register_0 = 0

    while True:
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        frame = cv2.flip(frame, 1)

        # Process the frame for hand detection
        frame = detector.findHands(frame)
        frame, dist = detector.getdistbetweenfingers(frame)
        if dist:
            print(f"distance between fingers: ({dist})")
            dist_vals.append(dist)
            ax_dist.plot(dist_vals, 'b')
            fig_dist.canvas.draw()
            fig_dist.canvas.flush_events()


        state = con.receive()
        if state is None:
            print("Failed to receive state")
            break

        x, y, z = detector.getIndexFingerMCPPosition()

        if x and y and z:
            x = x * 0.35 + 0.4
            y = y * 0.42 - 0.21

            data = [x, y, 0.01, 0, 3.11, 0.04, dist]
            if valable(x, y, z, robot_x_min, robot_x_max, robot_y_min, robot_y_max, robot_z_min,
                       robot_z_max):
                print('data :', data)
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)
                scatter._offsets3d = (x_vals, y_vals, z_vals)
                fig.canvas.draw()
                fig.canvas.flush_events()
                list_to_setp(setp, data)
                con.send(setp)

                con.send(watchdog)

            else:
                print(f"you are not in the right zone")

        cv2.imshow('OAK-D RGB with Mediapipe', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    plt.show()
    cv2.destroyAllWindows()
    plt.ioff()
    con.send_pause()
    con.disconnect()
    device.close()


if __name__ == '__main__':
    main()
