"""
detection.py
Deteksi wajah dan pencarian KANDIDAT jerawat dengan metode klasik:
grayscale -> enhancement -> threshold -> morfologi -> contour -> filter.
Hasil hanya indikasi berdasarkan citra, BUKAN diagnosis medis.
"""

import cv2
import numpy as np

from image_processing import (
    resize_image, to_grayscale, reduce_noise, enhance_contrast, MIN_SIDE
)


def _ganjil(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def detect_face(gray):
    """Cari wajah dengan Haar Cascade. Mengembalikan (x, y, w, h) wajah
    terbesar, atau None jika tidak ada wajah."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    h, w = gray.shape
    ukuran_min = max(60, min(h, w) // 6)
    wajah = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(ukuran_min, ukuran_min))
    if len(wajah) == 0:
        return None
    x, y, fw, fh = max(wajah, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(fw), int(fh)


def buat_mask_kulit(face_bgr, buang_fitur=True):
    """Mask area kulit (putih = kulit) agar rambut, mata, dan bibir
    tidak dianggap jerawat. Memakai ruang warna YCrCb."""
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    # Closing besar: menutup lubang kecil supaya bintik merah (jerawat)
    # yang warnanya di luar rentang kulit tidak ikut terbuang.
    h, w = mask.shape
    k = _ganjil(max(5, w // 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    # Buang tepi kotak wajah (biasanya rambut / background)
    tepi = np.zeros_like(mask)
    tepi[int(0.05 * h):int(0.97 * h), int(0.10 * w):int(0.90 * w)] = 255
    mask = cv2.bitwise_and(mask, tepi)

    if buang_fitur:
        # Perkiraan posisi mata kiri, mata kanan, dan mulut di kotak wajah
        for (x1, y1, x2, y2) in [(0.10, 0.22, 0.45, 0.45),
                                 (0.55, 0.22, 0.90, 0.45),
                                 (0.28, 0.68, 0.72, 0.88)]:
            cv2.rectangle(mask, (int(x1 * w), int(y1 * h)),
                          (int(x2 * w), int(y2 * h)), 0, -1)
    return mask


def buat_peta_skor(face_bgr, face_enh):
    """Peta 'seberapa mencurigakan' tiap piksel, dibandingkan dengan kulit
    di sekitarnya (bukan dengan nilai global). Karena itu lebih tahan
    terhadap perbedaan pencahayaan, mirip adaptive threshold.
      - kemerahan : kanal a* (Lab) lebih tinggi dari sekitarnya
      - kegelapan : intensitas lebih rendah dari sekitarnya
    """
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    g = face_enh.astype(np.float32)

    k = _ganjil(max(15, face_bgr.shape[1] // 4))
    latar_a = cv2.GaussianBlur(a, (k, k), 0)   # rata-rata lokal kemerahan
    latar_g = cv2.GaussianBlur(g, (k, k), 0)   # rata-rata lokal intensitas

    kemerahan = a - latar_a
    kegelapan = latar_g - g
    skor = 1.5 * kemerahan + 0.5 * kegelapan
    return np.clip(skor, 0, 255).astype(np.uint8)


def deskripsi_lokasi(rx, ry):
    """Ubah posisi relatif (0-1) di dalam wajah menjadi teks lokasi."""
    if ry < 0.30:
        vertikal = "dahi"
    elif ry < 0.70:
        vertikal = "pipi/hidung"
    else:
        vertikal = "sekitar mulut/dagu"
    if rx < 0.35:
        horizontal = "sisi kiri gambar"
    elif rx > 0.65:
        horizontal = "sisi kanan gambar"
    else:
        horizontal = "tengah"
    return f"{vertikal}, {horizontal}"


def analyze_image(img_bgr, params):
    """Pipeline lengkap. Mengembalikan dict berisi semua gambar hasil.
    Tidak melempar error untuk kasus umum (gambar kecil, bukan wajah, dll)."""
    hasil = {
        "gambar_asli": img_bgr, "gray": None, "enhanced": None,
        "segmentasi": None, "kontur": None, "deteksi": None,
        "wajah_ditemukan": False, "kandidat": [],
        "error": None, "peringatan": None,
    }

    # --- Cek ukuran gambar ---
    if min(img_bgr.shape[:2]) < MIN_SIDE:
        hasil["error"] = (f"Gambar terlalu kecil (minimal {MIN_SIDE} piksel "
                          "pada sisi terpendek). Gunakan foto yang lebih besar.")
        return hasil

    # --- 1-4. Resize, grayscale ---
    img = resize_image(img_bgr)
    hasil["gambar_asli"] = img
    gray = to_grayscale(img)
    hasil["gray"] = gray

    # --- 5-6. Reduksi noise + contrast enhancement ---
    img_blur = reduce_noise(img, params["blur_metode"], params["blur_ksize"])
    gray_blur = to_grayscale(img_blur)
    enhanced = enhance_contrast(gray_blur, params["contrast"])
    hasil["enhanced"] = enhanced

    # --- 7. Deteksi wajah ---
    box = detect_face(enhanced)
    fallback = False
    if box is None:
        if params["fallback"]:
            box = (0, 0, img.shape[1], img.shape[0])
            fallback = True
            hasil["peringatan"] = ("Wajah tidak terdeteksi, seluruh gambar "
                                   "dianalisis. Hasil kurang bisa diandalkan.")
        else:
            hasil["error"] = ("Wajah tidak terdeteksi. Gunakan foto wajah "
                              "menghadap depan dengan cahaya cukup.")
            return hasil
    hasil["wajah_ditemukan"] = True

    # --- 8. Batasi analisis di area wajah ---
    fx, fy, fw, fh = box
    face_bgr = img_blur[fy:fy + fh, fx:fx + fw]
    face_enh = enhanced[fy:fy + fh, fx:fx + fw]
    kulit = buat_mask_kulit(face_bgr, params["buang_fitur"] and not fallback)

    # --- 9. Thresholding -> segmentasi ---
    skor = buat_peta_skor(face_bgr, face_enh)
    # Sensitivity tinggi = ambang lebih rendah = lebih banyak kandidat
    ambang = params["threshold"] * (1.5 - params["sensitivity"] / 10)
    _, biner = cv2.threshold(skor, ambang, 255, cv2.THRESH_BINARY)
    biner = cv2.bitwise_and(biner, kulit)

    # --- 10. Morfologi: opening (hapus bintik noise), closing (tutup lubang) ---
    biner = cv2.morphologyEx(biner, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    biner = cv2.morphologyEx(biner, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    seg = np.zeros(gray.shape, np.uint8)
    seg[fy:fy + fh, fx:fx + fw] = biner
    hasil["segmentasi"] = seg

    # --- 11. Cari contour ---
    kontur_semua, _ = cv2.findContours(
        biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 12. Filter kandidat ---
    diterima = []
    for c in kontur_semua:
        luas = cv2.contourArea(c)
        if luas < params["min_area"] or luas > params["max_area"]:
            continue                                   # terlalu kecil / besar
        x, y, w, h = cv2.boundingRect(c)
        rasio = w / h
        if rasio < 0.4 or rasio > 2.5:
            continue                                   # terlalu memanjang
        keliling = cv2.arcLength(c, True)
        if keliling == 0:
            continue
        kebulatan = 4 * np.pi * luas / (keliling ** 2)  # 1.0 = lingkaran
        if kebulatan < 0.3:
            continue                                   # bentuk tidak beraturan
        mask_c = np.zeros(biner.shape, np.uint8)
        cv2.drawContours(mask_c, [c], -1, 255, -1)
        rata_skor = cv2.mean(skor, mask=mask_c)[0]
        if rata_skor < ambang * 1.1:
            continue                                   # kurang merah/gelap
        diterima.append((c, x, y, w, h, luas, kebulatan, rata_skor))

    # --- 13-14. Gambar contour & bounding box, hitung kandidat ---
    gambar_kontur = img.copy()
    cv2.drawContours(gambar_kontur, kontur_semua, -1, (255, 0, 0), 1,
                     offset=(fx, fy))          # biru = semua contour
    gambar_deteksi = img.copy()
    cv2.rectangle(gambar_deteksi, (fx, fy), (fx + fw, fy + fh), (255, 128, 0), 2)

    for i, (c, x, y, w, h, luas, kebulatan, rata_skor) in enumerate(diterima, 1):
        cv2.drawContours(gambar_kontur, [c], -1, (0, 255, 0), 2,
                         offset=(fx, fy))      # hijau = lolos filter
        gx, gy = x + fx, y + fy
        cv2.rectangle(gambar_deteksi, (gx, gy), (gx + w, gy + h), (0, 0, 255), 2)
        cv2.putText(gambar_deteksi, str(i), (gx, max(gy - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        hasil["kandidat"].append({
            "No": i,
            "Lokasi": deskripsi_lokasi((x + w / 2) / fw, (y + h / 2) / fh),
            "Posisi (x, y)": f"({gx}, {gy})",
            "Ukuran (px)": f"{w} x {h}",
            "Luas (px²)": int(luas),
            "Kebulatan": round(float(kebulatan), 2),
            "Skor": round(float(rata_skor), 1),
        })

    hasil["kontur"] = gambar_kontur
    hasil["deteksi"] = gambar_deteksi
    return hasil