import cv2
import numpy as np

atha = cv2.VideoCapture("drift.mp4")

while True:
    ret, frame = atha.read()
    if not ret:
        break

    f1 = frame.copy()

    f2 = frame.copy()
    f2[:, :, 0] = 0 
    f2[:, :, 1] = 0 

    f3 = frame.copy()
    f3[:, :, 0] = 0 
    f3[:, :, 2] = 0 

    f4 = frame.copy()
    f4[:, :, 1] = 0
    f4[:, :, 2] = 0

    h, w = frame.shape[:2]
    f1 = cv2.resize(f1, (w//4, h//4))
    f2 = cv2.resize(f2, (w//4, h//4))
    f3 = cv2.resize(f3, (w//4, h//4))
    f4 = cv2.resize(f4, (w//4, h//4))

    top    = np.hstack((f1, f2))
    bottom = np.hstack((f3, f4))
    combined = np.vstack((top, bottom))

    cv2.imshow("4 Filters", combined)

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

atha.release()
cv2.destroyAllWindows()