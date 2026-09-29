import cv2
import numpy as np

def original(img):
    return img.copy()

def red(img):
    out = np.zeros_like(img)
    out[:, :, 2] = img[:, :, 2]
    return out

def green(img):
    out = np.zeros_like(img)
    out[:, :, 1] = img[:, :, 1]
    return out

def blue(img):
    out = np.zeros_like(img)
    out[:, :, 0] = img[:, :, 0]
    return out

filters = [original, red, green, blue]

img = cv2.imread("crazy.jpeg")

tiles = [fn(img) for fn in filters]

top = np.hstack(tiles[:2])
bottom = np.hstack(tiles[2:])
combined = np.vstack((top, bottom))

h, w = combined.shape[:2]
combined = cv2.resize(combined, (w // 2, h // 4))

cv2.imshow("RGB Channels", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()