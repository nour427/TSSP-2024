# THE COMPUTER VISION PART ONLY
import cv2
import mediapipe as mp
from examples.CamSetup import setup_camera
import matplotlib.pyplot as plt
import math
from collections import deque
import numpy as np


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
def valable(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax

def scale(pos, min, max, robotmin, robotmax):
    return robotmin + (pos - min) * (robotmax - robotmin) / (max - min)

def main():
    device, qRgb = setup_camera()  # Get the OAK-D camera setup
    detector = HandDetector()
    # Initialize OpenCV window for camera feed
    cv2.namedWindow('OAK-D RGB with Mediapipe')

    # Initialize matplotlib figure and scatter plot
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

    x_vals = []
    y_vals = []
    z_vals = []

    robot_x_max = 0.75
    robot_x_min = 0.4
    robot_y_max = 0.21
    robot_y_min = -0.21
    robot_z_max = 0.3
    robot_z_min = 0.003

    buffer_size = 10
    x_buffer = deque(maxlen=buffer_size)
    y_buffer = deque(maxlen=buffer_size)
    orientation_buffer = deque(maxlen=buffer_size)
    dist_buffer = deque(maxlen=buffer_size)
    #bounds = (0.400, 0.750, -0.210, 0.210, 0.003, 0.300)
    bounds = (-1, 1, -1, 1, -1, 1)


    while True:
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        frame = cv2.flip(frame, 1)


        # Process the frame for hand detection
        frame = detector.findHands(frame)
        frame, dist = detector.getdistbetweenfingers(frame)


        x, y, z = detector.getIndexFingerMCPPosition()
        #orientation_x, orientation_y, orientation_z = detector.getOrientation()
        angle = detector.getOrientation()
        if dist:
            print(f"distance between fingers: ({dist})")
            dist_vals.append(z)
            ax_dist.plot(dist_vals, 'b')
            fig_dist.canvas.draw()
            fig_dist.canvas.flush_events()
            dist_buffer.append(dist)






            avg_dist = np.mean(dist_buffer, axis=0)
            cv2.putText(frame, f"distance : {dist}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.putText(frame, f"Avg distance : {avg_dist}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        if x and y and z:
            if valable(x, y, z, bounds):
                x = scale(x, 0.97, 0.04, robot_x_min, robot_x_max)
                y = scale(y, 0.06, 0.9, robot_y_min, robot_y_max)
                print(f"Index Finger MCP Position: ({x}, {y}, {z})")
                x_buffer.append(x)
                y_buffer.append(y)
                avg_x = np.mean(x_buffer, axis=0)
                avg_y = np.mean(y_buffer, axis=0)

                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)
                scatter._offsets3d = (x_vals, y_vals, z_vals)
                fig.canvas.draw()
                fig.canvas.flush_events()
                cv2.putText(frame, f"x: {x}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"y: {y}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (
255, 0, 0), 2)
                cv2.putText(frame, f"Avg x: {avg_x}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Avg y: {avg_y}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        #if orientation_x is not None and orientation_y is not None and orientation_z is not None:
        if angle is not None:

            angle =-angle + math.pi/2
            orientation_buffer.append(angle)
            avg_angle = np.mean(orientation_buffer, axis=0)
            Rx = math.sin(angle) * math.pi
            Ry = math.cos(angle) * math.pi

            cv2.putText(frame, f"Orientation : {angle}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            cv2.putText(frame, f"Avg Orientation : {avg_angle}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)




            # cv2.putText(frame, f"Orientation Z: {orientation_z:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)

        else:

            print("You are not in the right zone")





        # Resize for better display
        cv2.imshow('OAK-D RGB with Mediapipe', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    plt.show()
    cv2.destroyAllWindows()
    # Turn off interactive mode
    plt.ioff()
    device.close()



if __name__ == '__main__':
    main()
