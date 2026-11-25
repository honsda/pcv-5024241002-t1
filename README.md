# Praktikum Pengolahan Citra Digital

Kumpulan tugas pengolahan citra digital dengan Python, OpenCV, dan NumPy.
Setiap folder berisi satu tugas beserta citra uji dan hasilnya.

| Folder | File | Materi | Dokumentasi |
|--------|------|--------|-------------|
| `t1` | `1-Intro.py` | Read image, show image, filter warna gambar (R/G/B), filter warna video | [README](t1/README.md) |
| `t2` | `2-ti-eq.py` | Transformasi intensitas + ekualisasi histogram (manual) | [README](t2/README.md) |
| `t3` | `3-filter-spasial.py` | Mean, Gaussian, median, sharpening, Sobel | [README](t3/README.md) |
| `t4` | `4-model-warna.py` | Konversi RGB, CMYK, HSI, HSV | [README](t4/README.md) |
| `t5` | `5-morphology.py` | Erosi, dilasi, opening, closing | [README](t5/README.md) |

## Menjalankan

```bash
cd first
python t1/1-Intro.py
python t2/2-ti-eq.py
python t3/3-filter-spasial.py
python t4/4-model-warna.py
python t5/5-morphology.py
```

Semua path di dalam script memakai lokasi file script, jadi script juga bisa
dijalankan langsung dari folder tugasnya.

## Kebutuhan

```bash
pip install opencv-python numpy matplotlib
```

## Catatan

- `t1/introimg.py` dan `t1/intro.py` adalah versi terpisah (gambar dan video).
  `1-Intro.py` menggabungkan keduanya.
- `t2/intensity.py` adalah versi lama tanpa ekualisasi histogram.
  `2-ti-eq.py` sudah menambahkan ekualisasi histogram.
- Ekualisasi histogram pada `t2` dihitung manual (tanpa `cv2.equalizeHist`),
  begitu juga filter spasial dan morfologi (tanpa `cv2.filter2D`, `cv2.erode`,
  dan sejenisnya).
- Hasil tiap tugas disimpan sebagai PNG terpisah per efek dengan ukuran kecil
  di folder tugasnya, dan ditampilkan di README masing-masing.
- Tekan tombol apa saja untuk menutup jendela gambar, dan `q` untuk
  menghentikan video.
