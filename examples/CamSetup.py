# SetUp of the camera
import cv2
import depthai as dai
#depth ai is a Spatial AI platform that is used for communication with and development of our devices; OAK cameras and RAE robots.
#DepthAI is the Embedded, Performant, Spatial AI+CV platform, composed of an open-source hardware, firmware,
# software ecosystem that provides turnkey embedded Spatial AI+CV and hardware-accelerated computer vision.
#depth ai helps us to access the camera and facilitates the processing of computer vision
#this function initilizes the camera
#I used a pipline which is a core concept of depth ai , it allows us to manage
# multiple streams of data and easily set up complex workflows
def setup_camera():
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)

    # Linking
    camRgb.video.link(xoutRgb.input)

    # Connect to device and start pipeline
    device = dai.Device(pipeline)

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    return device, qRgb

#this function displays the content of the camera
def show_camera_feed():
    device, qRgb = setup_camera()

    while True:
        inRgb = qRgb.get()  # Blocking call, will wait until a new data has arrived
        frame = inRgb.getCvFrame()

        # Show the frame
        cv2.imshow("OAK-D RGB", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_camera_feed()