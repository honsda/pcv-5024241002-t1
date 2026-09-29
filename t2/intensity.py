import cv2
import numpy as np

img = cv2.imread("t2/crazy.png", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

neg = np.zeros((h, w), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        neg[i, j] = 255 - img[i, j]

c = 255 / np.log(1 + 255)
log_img = np.zeros((h, w), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        log_img[i, j] = int(c * np.log(1 + int(img[i, j])))

gamma = 0.5
gam = np.zeros((h, w), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        gam[i, j] = int(255 * ((img[i, j] / 255) ** gamma))

rmin, rmax = 255, 0
for i in range(h):
    for j in range(w):
        if img[i, j] < rmin: rmin = img[i, j]
        if img[i, j] > rmax: rmax = img[i, j]

stretch = np.zeros((h, w), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        stretch[i, j] = int((img[i, j] - rmin) / (rmax - rmin) * 255)

blank = np.zeros_like(img)
top = np.hstack((img, neg, log_img))
bottom = np.hstack((gam, stretch, blank))
combined = np.vstack((top, bottom))

ch, cw = combined.shape[:2]
combined = cv2.resize(combined, (cw // 4, ch // 4))

cv2.imshow("", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()