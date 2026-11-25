import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE, "crazy.png")


def histogram(img):
    hist = np.zeros(256, dtype=np.int64)
    for value in img.ravel():
        hist[value] += 1
    return hist


def cumulative(hist):
    cdf = np.zeros(256, dtype=np.int64)
    total = 0
    for k in range(256):
        total += int(hist[k])
        cdf[k] = total
    return cdf


def negative(img):
    return 255 - img


def log_transform(img):
    c = 255.0 / np.log(1.0 + 255.0)
    out = c * np.log(1.0 + img.astype(np.float64))
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_transform(img, gamma=0.5):
    out = 255.0 * (img.astype(np.float64) / 255.0) ** gamma
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast_stretch(img, hist):
    values = np.flatnonzero(hist)
    rmin, rmax = int(values[0]), int(values[-1])
    if rmax == rmin:
        return img.copy()
    out = (img.astype(np.float64) - rmin) * 255.0 / (rmax - rmin)
    return np.clip(out, 0, 255).astype(np.uint8)


def equalize_histogram(img, hist):
    height, width = img.shape
    total = height * width

    cdf = cumulative(hist)

    lut = np.zeros(256, dtype=np.uint8)
    for k in range(256):
        lut[k] = int(round(255.0 * cdf[k] / total))

    equalized = lut[img]
    return equalized, lut, cdf


def to_display(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def resize_width(img, width):
    out = to_display(img)
    h, w = out.shape[:2]
    return cv2.resize(out, (width, int(width * h / w)),
                      interpolation=cv2.INTER_AREA)


def show_result(title, filename, img, width=480):
    small = resize_width(img, width)
    cv2.imwrite(os.path.join(BASE, filename), small)
    cv2.imshow(title, small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_histogram(filename, title, values):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(np.arange(256), values, width=1.0, color="black")
    ax.set_title(title)
    ax.set_xlabel("Intensitas")
    ax.set_ylabel("Jumlah piksel")
    ax.set_xlim([0, 255])
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, filename), dpi=100)
    return fig


def save_cdf(filename, title, cdf, total):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(256), cdf / total, color="blue")
    ax.set_title(title)
    ax.set_xlabel("Intensitas")
    ax.set_ylabel("Probabilitas")
    ax.set_xlim([0, 255])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, filename), dpi=100)
    return fig


def main():
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {IMAGE_PATH}")

    hist = histogram(img)
    neg = negative(img)
    log_img = log_transform(img)
    gamma_img = gamma_transform(img, gamma=0.5)
    stretch = contrast_stretch(img, hist)
    eq, lut, cdf = equalize_histogram(img, hist)
    hist_eq = histogram(eq)
    cdf_eq = cumulative(hist_eq)

    def stats(h):
        total = int(h.sum())
        mean = float((np.arange(256) * h).sum()) / total
        low = int(np.flatnonzero(h)[0])
        high = int(np.flatnonzero(h)[-1])
        return low, high, mean

    print("Original  : min=%d max=%d mean=%.2f" % stats(hist))
    print("Ekualisasi: min=%d max=%d mean=%.2f" % stats(hist_eq))
    print("LUT equalisasi (0..255):")
    print(lut)

    show_result("Original", "hasil-original.png", img)
    show_result("Negative", "hasil-negative.png", neg)
    show_result("Log", "hasil-log.png", log_img)
    show_result("Gamma 0.5", "hasil-gamma.png", gamma_img)
    show_result("Contrast Stretch", "hasil-stretch.png", stretch)
    show_result("Histogram Equalization", "hasil-equalization.png", eq)

    save_histogram("hasil-hist-original.png", "Histogram Original", hist)
    save_histogram("hasil-hist-equalization.png",
                   "Histogram Setelah Ekualisasi", hist_eq)
    save_cdf("hasil-cdf-original.png", "CDF Original", cdf, img.size)
    save_cdf("hasil-cdf-equalization.png",
             "CDF Setelah Ekualisasi", cdf_eq, img.size)
    plt.show()


if __name__ == "__main__":
    main()
