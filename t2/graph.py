import cv2
from matplotlib import pyplot as plt

img = cv2.imread("t2/crazy.png")

cnv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(cnv)

plt.figure(figsize=(10, 6))

plt.hist(r.ravel(), bins=256, color='red', alpha=0.5, label='Red Channel')
plt.hist(g.ravel(), bins=256, color='green', alpha=0.5, label='Green Channel')
plt.hist(b.ravel(), bins=256, color='blue', alpha=0.5, label='Blue Channel')

plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.show()