# Tugas 4 - Model Warna

Konversi model warna RGB ke CMYK, HSI, dan HSV beserta konversi baliknya. Semua rumus dihitung manual tanpa `cv2.cvtColor`. Setiap kanal dan hasil rekonstruksi disimpan sebagai gambar kecil terpisah.

## Kode

```python
import cv2
import numpy as np

def rgb_to_cmyk(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    k = 1.0 - np.max(rgb, axis=2)

    denom = 1.0 - k
    safe = np.where(denom > 0, denom, 1.0)
    c = (1.0 - r - k) / safe
    m = (1.0 - g - k) / safe
    y = (1.0 - b - k) / safe

    black = denom <= 0
    c[black] = 0.0
    m[black] = 0.0
    y[black] = 0.0
    return np.stack((c, m, y, k), axis=2)

def cmyk_to_rgb(cmyk):
    c, m, y, k = (cmyk[:, :, i] for i in range(4))
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return np.stack((r, g, b), axis=2)

def rgb_to_hsi(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    sumc = r + g + b
    minc = np.min(rgb, axis=2)

    intensity = sumc / 3.0
    safe_sum = np.where(sumc > 0, sumc, 1.0)
    saturation = np.where(sumc > 0, 1.0 - 3.0 * minc / safe_sum, 0.0)

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    safe_den = np.where(den > 0, den, 1.0)
    theta = np.arccos(np.clip(num / safe_den, -1.0, 1.0))
    hue = np.degrees(theta)
    hue = np.where(b > g, 360.0 - hue, hue)
    hue = np.where(den > 0, hue, 0.0)

    return np.stack((hue / 360.0, saturation, intensity), axis=2)

def hsi_to_bgr(h, s, i):
    hue = np.mod(h, 1.0) * 360.0
    r = np.zeros_like(hue)
    g = np.zeros_like(hue)
    b = np.zeros_like(hue)

    m = (hue >= 0) & (hue < 120)
    if np.any(m):
        hh = np.deg2rad(hue[m])
        base, sat = i[m], s[m]
        b0 = base * (1.0 - sat)
        r0 = base * (1.0 + sat * np.cos(hh) / np.cos(np.deg2rad(60.0) - hh))
        g0 = 3.0 * base - (r0 + b0)
        r[m], g[m], b[m] = r0, g0, b0

    m = (hue >= 120) & (hue < 240)
    if np.any(m):
        hh = np.deg2rad(hue[m] - 120.0)
        base, sat = i[m], s[m]
        r0 = base * (1.0 - sat)
        g0 = base * (1.0 + sat * np.cos(hh) / np.cos(np.deg2rad(60.0) - hh))
        b0 = 3.0 * base - (r0 + g0)
        r[m], g[m], b[m] = r0, g0, b0

    m = (hue >= 240) & (hue < 360)
    if np.any(m):
        hh = np.deg2rad(hue[m] - 240.0)
        base, sat = i[m], s[m]
        g0 = base * (1.0 - sat)
        b0 = base * (1.0 + sat * np.cos(hh) / np.cos(np.deg2rad(60.0) - hh))
        r0 = 3.0 * base - (g0 + b0)
        r[m], g[m], b[m] = r0, g0, b0

    return np.clip(np.stack((b, g, r), axis=2), 0.0, 1.0)

def rgb_to_hsv(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    delta = maxc - minc

    safe_max = np.where(maxc > 0, maxc, 1.0)
    safe_delta = np.where(delta > 0, delta, 1.0)

    value = maxc
    saturation = np.where(maxc > 0, delta / safe_max, 0.0)

    h_r = 60.0 * (((g - b) / safe_delta) % 6.0)
    h_g = 60.0 * ((b - r) / safe_delta + 2.0)
    h_b = 60.0 * ((r - g) / safe_delta + 4.0)

    hue = np.where(maxc == r, h_r,
                   np.where(maxc == g, h_g,
                            np.where(maxc == b, h_b, 0.0)))
    hue = np.where(delta > 0, hue, 0.0)
    hue = np.where(hue < 0, hue + 360.0, hue)

    return np.stack((hue / 360.0, saturation, value), axis=2)

def hsv_to_bgr(h, s, v):
    h6 = np.mod(h, 1.0) * 6.0
    sector = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.select([sector == 0, sector == 1, sector == 2,
                   sector == 3, sector == 4, sector == 5],
                  [v, q, p, p, t, v])
    g = np.select([sector == 0, sector == 1, sector == 2,
                   sector == 3, sector == 4, sector == 5],
                  [t, v, v, q, p, p])
    b = np.select([sector == 0, sector == 1, sector == 2,
                   sector == 3, sector == 4, sector == 5],
                  [p, p, t, v, v, q])
    return np.clip(np.stack((b, g, r), axis=2), 0.0, 1.0)
```

Kode di atas hanya bagian pengolahan citra. Bagian pembacaan, penampilan, dan penyimpanan hasil beserta nama file output tidak ditampilkan. Kode lengkap ada di `4-model-warna.py`.

## CMYK

`K = 1 - max(R,G,B)`, `C = (1-R-K)/(1-K)`, begitu pula M dan Y.

### Original (RGB)

![Original (RGB)](./hasil-cmyk-original.png)

### Cyan

![Cyan](./hasil-cmyk-c.png)

### Magenta

![Magenta](./hasil-cmyk-m.png)

### Yellow

![Yellow](./hasil-cmyk-y.png)

### Key / Black

![Key / Black](./hasil-cmyk-k.png)

### CMYK -> RGB

![CMYK -> RGB](./hasil-cmyk-rgb.png)

## HSI

`I = (R+G+B)/3`, `S = 1 - 3min(R,G,B)/(R+G+B)`, hue memakai rumus cosinus. Hue ditampilkan berwarna agar jenis warnanya terlihat.

### Original (RGB)

![Original (RGB)](./hasil-hsi-original.png)

### Hue

![Hue](./hasil-hsi-h.png)

### Saturation

![Saturation](./hasil-hsi-s.png)

### Intensity

![Intensity](./hasil-hsi-i.png)

### HSI -> RGB

![HSI -> RGB](./hasil-hsi-rgb.png)

## HSV

`V = max(R,G,B)`, `S = (max-min)/max`, hue berdasarkan kanal maksimum. Konversi balik HSV ke RGB menghasilkan citra yang sama.

### Original (RGB)

![Original (RGB)](./hasil-hsv-original.png)

### Hue

![Hue](./hasil-hsv-h.png)

### Saturation

![Saturation](./hasil-hsv-s.png)

### Value

![Value](./hasil-hsv-v.png)

### HSV -> RGB

![HSV -> RGB](./hasil-hsv-rgb.png)
