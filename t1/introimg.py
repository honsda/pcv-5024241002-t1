import cv2
import numpy as np

def original(img):
    return img.copy()

def red(img):
    out = img.copy()
    out[:, :, 0] = 0
    out[:, :, 1] = 0
    return out

def green(img):
    out = img.copy()
    out[:, :, 0] = 0
    out[:, :, 2] = 0
    return out

def blue(img):
    out = img.copy()
    out[:, :, 1] = 0
    out[:, :, 2] = 0
    return out

filters = [original, red, green, blue]

img = cv2.imread("t1/crazy.jpeg")

tiles = [fn(img) for fn in filters]

top = np.hstack(tiles[:2])
bottom = np.hstack(tiles[2:])
combined = np.vstack((top, bottom))

h, w = combined.shape[:2]
combined = cv2.resize(combined, (w // 2, h // 4))

cv2.imshow("", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()