import cv2
from examples.CamSetup import setup_camera

device, qRgb = setup_camera()

num = 0

while True:

    inRgb = qRgb.get()
    frame = inRgb.getCvFrame()
    height, width = frame.shape[:2]
    #print(height,'  ', width)
    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('images/img' + str(num) + '.png', frame)
        print("image saved!")
        num += 1

    cv2.imshow('Img', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release and destroy all windows before termination
cv2.destroyAllWindows()
device.close()
