"""
image_processing.py
Fungsi-fungsi dasar pengolahan citra (sesuai materi praktikum ImageJ):
RGB -> grayscale, histogram, reduksi noise, contrast enhancement.
Tidak ada kode Streamlit di file ini.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend tanpa jendela, aman untuk Streamlit
import matplotlib.pyplot as plt

MAX_SIDE = 800   # sisi terpanjang gambar dibatasi agar proses cepat & parameter stabil
MIN_SIDE = 200   # gambar lebih kecil dari ini dianggap terlalu kecil


def decode_image(file_bytes):
    """Membaca byte file upload menjadi gambar BGR (OpenCV).
    Mengembalikan None jika file bukan gambar yang valid."""
    data = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_image(img, max_side=MAX_SIDE):
    """Perkecil gambar jika terlalu besar (rasio tetap)."""
    h, w = img.shape[:2]
    skala = max_side / max(h, w)
    if skala >= 1:
        return img
    return cv2.resize(img, (int(w * skala), int(h * skala)),
                      interpolation=cv2.INTER_AREA)


def to_grayscale(img_bgr):
    """Ubah citra berwarna (3 kanal) menjadi grayscale (1 kanal intensitas 0-255)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def make_histogram_figure(gray, judul="Histogram Grayscale"):
    """Membuat histogram distribusi intensitas piksel (0-255)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(256), hist, width=1.0, color="dimgray")
    ax.set_xlim(0, 255)
    ax.set_title(judul)
    ax.set_xlabel("Intensitas piksel (0 = hitam, 255 = putih)")
    ax.set_ylabel("Jumlah piksel")
    fig.tight_layout()
    return fig


def reduce_noise(img, metode="Gaussian", ksize=5):
    """Kurangi noise dengan Gaussian blur atau median filter.
    ksize harus ganjil. Bisa dipakai untuk gambar berwarna maupun grayscale."""
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    if metode == "Median":
        return cv2.medianBlur(img, ksize)
    if metode == "Gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img  # "Tidak ada"


def enhance_contrast(gray, contrast=2.0):
    """Contrast enhancement memakai CLAHE (histogram equalization lokal).
    contrast = clip limit; makin besar makin kuat kontrasnya."""
    clahe = cv2.createCLAHE(clipLimit=max(float(contrast), 0.1),
                            tileGridSize=(8, 8))
    return clahe.apply(gray)