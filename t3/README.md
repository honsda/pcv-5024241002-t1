# Tugas 3 - Filter Spasial

Menerapkan filter spasial secara manual tanpa `cv2.filter2D`, `cv2.blur`, `cv2.GaussianBlur`, atau `cv2.medianBlur`. Setiap filter disimpan sebagai gambar kecil terpisah.

## Kode

```python
import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE, "crazy.png")

MEAN_5 = np.ones((5, 5), dtype=np.float32) / 25.0
SHARPEN = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]], dtype=np.float32)
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float32)


def gaussian_kernel(size=5, sigma=1.0):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


GAUSS_5 = gaussian_kernel(5, 1.5)


def convolve_float(img, kernel):
    single = img.ndim == 2
    src = img[:, :, None] if single else img
    h, w, c = src.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(src.astype(np.float32), ((ph, ph), (pw, pw), (0, 0)),
                    mode="edge")
    out = np.zeros((h, w, c), dtype=np.float32)

    for i in range(kh):
        for j in range(kw):
            weight = kernel[i, j]
            if weight != 0:
                out += weight * padded[i:i + h, j:j + w, :]

    return out[:, :, 0] if single else out


def convolve(img, kernel):
    out = convolve_float(img, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def sobel_edges(img, gain=3.0):
    if img.ndim == 3:
        gray = (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1]
                + 0.299 * img[:, :, 2]).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    gx = convolve_float(gray, SOBEL_X)
    gy = convolve_float(gray, SOBEL_Y)
    magnitude = np.sqrt(gx * gx + gy * gy)
    return np.clip(magnitude * gain, 0, 255).astype(np.uint8)


def median_filter_gray(img, size=3):
    pad = size // 2
    h, w = img.shape
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")

    windows = [padded[i:i + h, j:j + w]
               for i in range(size) for j in range(size)]
    return np.median(np.stack(windows, axis=0), axis=0).astype(np.uint8)


def median_filter(img, size=3):
    if img.ndim == 2:
        return median_filter_gray(img, size)
    return np.stack([median_filter_gray(img[:, :, ch], size)
                     for ch in range(img.shape[2])], axis=2)


def add_salt_pepper(img, amount=0.03, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = img.shape[:2]
    n = int(h * w * amount)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    half = n // 2
    out[ys[:half], xs[:half]] = 255
    out[ys[half:], xs[half:]] = 0
    return out


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


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {IMAGE_PATH}")

    blurred = convolve(img, MEAN_5)
    gaussian = convolve(img, GAUSS_5)
    median = median_filter(img, size=3)
    sharpened = convolve(img, SHARPEN)
    edges = sobel_edges(img, gain=3.0)

    show_result("Original", "hasil-original.png", img)
    show_result("Mean 5x5", "hasil-mean.png", blurred)
    show_result("Gaussian 5x5", "hasil-gaussian.png", gaussian)
    show_result("Median 3x3", "hasil-median.png", median)
    show_result("Sharpening 3x3", "hasil-sharpen.png", sharpened)
    show_result("Sobel (edge)", "hasil-sobel.png", edges)

    y0, y1, x0, x1 = 200, 500, 900, 1200
    show_result("Original (zoom)", "hasil-zoom-original.png",
                img[y0:y1, x0:x1], 300)
    show_result("Mean 5x5 (zoom)", "hasil-zoom-mean.png",
                blurred[y0:y1, x0:x1], 300)
    show_result("Gaussian 5x5 (zoom)", "hasil-zoom-gaussian.png",
                gaussian[y0:y1, x0:x1], 300)
    show_result("Median 3x3 (zoom)", "hasil-zoom-median.png",
                median[y0:y1, x0:x1], 300)

    noisy = add_salt_pepper(img, amount=0.03, seed=0)
    noisy_mean = convolve(noisy, MEAN_5)
    noisy_median = median_filter(noisy, size=3)

    show_result("Noisy (salt & pepper)", "hasil-noise.png", noisy)
    show_result("Noisy + Mean 5x5", "hasil-noise-mean.png", noisy_mean)
    show_result("Noisy + Median 3x3", "hasil-noise-median.png", noisy_median)


if __name__ == "__main__":
    main()
```

## Hasil Filter Spasial

### Original

![Original](./hasil-original.png)

### Mean 5x5

![Mean 5x5](./hasil-mean.png)

### Gaussian 5x5

![Gaussian 5x5](./hasil-gaussian.png)

### Median 3x3

![Median 3x3](./hasil-median.png)

### Sharpening 3x3

![Sharpening 3x3](./hasil-sharpen.png)

### Sobel (edge)

![Sobel (edge)](./hasil-sobel.png)

## Perbandingan Zoom 300x300

Area diperbesar agar efek penghalusan terlihat jelas. Mean dan Gaussian meratakan piksel, sedangkan median mempertahankan tepi objek.

### Original (zoom)

![Original (zoom)](./hasil-zoom-original.png)

### Mean 5x5 (zoom)

![Mean 5x5 (zoom)](./hasil-zoom-mean.png)

### Gaussian 5x5 (zoom)

![Gaussian 5x5 (zoom)](./hasil-zoom-gaussian.png)

### Median 3x3 (zoom)

![Median 3x3 (zoom)](./hasil-zoom-median.png)

## Mean vs Median pada Noise

Median filter lebih baik menghilangkan noise salt & pepper karena tidak mengambil rata-rata nilai ekstrem.

### Noisy (salt & pepper)

![Noisy (salt & pepper)](./hasil-noise.png)

### Noisy + Mean 5x5

![Noisy + Mean 5x5](./hasil-noise-mean.png)

### Noisy + Median 3x3

![Noisy + Median 3x3](./hasil-noise-median.png)
