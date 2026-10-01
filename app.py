"""
app.py
Antarmuka Streamlit. Semua pengolahan citra ada di image_processing.py
dan detection.py; file ini hanya mengatur tampilan.
"""

import cv2
import matplotlib.pyplot as plt
import streamlit as st

from image_processing import decode_image, make_histogram_figure
from detection import analyze_image

st.set_page_config(page_title="Indikasi Jerawat - Pengolahan Citra", layout="wide")
st.title("🔍 Analisis Kandidat Jerawat (Pengolahan Citra Digital)")
st.warning("Aplikasi ini hanya analisis berdasarkan karakteristik citra. "
           "Hasilnya BUKAN diagnosis medis.")

# ---------------- Sidebar ----------------
st.sidebar.header("Input")
file_upload = st.sidebar.file_uploader("Upload Foto", type=["jpg", "jpeg", "png"])

st.sidebar.header("Parameter")
blur_metode = st.sidebar.selectbox("Metode reduksi noise",
                                   ["Gaussian", "Median", "Tidak ada"])
blur_ksize = st.sidebar.slider("Ukuran kernel blur", 3, 15, 5, step=2)
contrast = st.sidebar.slider("Contrast (CLAHE)", 1.0, 6.0, 2.0, step=0.5)
threshold = st.sidebar.slider("Threshold", 5, 60, 20)
sensitivity = st.sidebar.slider("Sensitivity", 1, 10, 5,
                                help="Makin tinggi = makin banyak kandidat")
min_area = st.sidebar.slider("Minimum area (px²)", 5, 500, 15)
max_area = st.sidebar.slider("Maximum area (px²)", 100, 3000, 500)
buang_fitur = st.sidebar.checkbox("Abaikan area mata & mulut", value=True)
fallback = st.sidebar.checkbox("Analisis seluruh gambar jika wajah tidak terdeteksi",
                               value=False)

tombol = st.sidebar.button("Analisis Citra", type="primary")

# ---------------- Validasi input ----------------
if file_upload is None:
    st.info("Silakan upload foto wajah (jpg/png) melalui sidebar.")
    st.stop()

gambar = decode_image(file_upload.getvalue())
if gambar is None:
    st.error("File tidak bisa dibaca sebagai gambar.")
    st.stop()

if tombol:
    params = dict(blur_metode=blur_metode, blur_ksize=blur_ksize,
                  contrast=contrast, threshold=threshold,
                  sensitivity=sensitivity, min_area=min_area,
                  max_area=max_area, buang_fitur=buang_fitur,
                  fallback=fallback)
    st.session_state["hasil"] = analyze_image(gambar, params)
    st.session_state["nama_file"] = file_upload.name

if st.session_state.get("nama_file") != file_upload.name or "hasil" not in st.session_state:
    st.image(gambar, channels="BGR", caption="Foto yang diupload")
    st.info("Atur parameter di sidebar lalu klik **Analisis Citra**.")
    st.stop()

hasil = st.session_state["hasil"]

# ---------------- Pesan error / peringatan ----------------
if hasil["error"]:
    st.error(hasil["error"])
if hasil["peringatan"]:
    st.warning(hasil["peringatan"])


def tampil(judul, gambar_hasil, penjelasan, **kw):
    st.subheader(judul)
    st.image(gambar_hasil, **kw)
    with st.expander("Penjelasan"):
        st.write(penjelasan)


# ---------------- 1-2. Original & Grayscale ----------------
if hasil["gray"] is not None:
    kol1, kol2 = st.columns(2)
    with kol1:
        tampil("1. Foto Asli", hasil["gambar_asli"],
               "Citra RGB: tiap piksel punya 3 nilai (R, G, B).", channels="BGR")
    with kol2:
        tampil("2. Grayscale", hasil["gray"],
               "Grayscale mengubah 3 kanal warna menjadi 1 kanal intensitas "
               "(0-255). Proses jadi lebih sederhana dan fokus pada terang-gelap.")

    # ---------------- 3-4. Histogram & Enhancement ----------------
    st.subheader("3. Histogram Grayscale")
    kol3, kol4 = st.columns(2)
    with kol3:
        fig = make_histogram_figure(hasil["gray"], "Sebelum enhancement")
        st.pyplot(fig)
        plt.close(fig)
    with kol4:
        fig = make_histogram_figure(hasil["enhanced"], "Sesudah blur + kontras")
        st.pyplot(fig)
        plt.close(fig)
    with st.expander("Penjelasan"):
        st.write("Histogram menunjukkan sebaran intensitas piksel. Histogram "
                 "yang menumpuk di satu sisi berarti gambar terlalu gelap/terang "
                 "atau kontrasnya rendah. Setelah enhancement, sebaran biasanya "
                 "lebih merata.")

    tampil("4. Hasil Peningkatan Kontras", hasil["enhanced"],
           "Blur mengurangi noise, lalu CLAHE meningkatkan kontras lokal "
           "agar perbedaan antara kulit dan bercak lebih jelas.")

# ---------------- 5-7. Segmentasi, Contour, Deteksi ----------------
if hasil["wajah_ditemukan"]:
    kol5, kol6 = st.columns(2)
    with kol5:
        tampil("5. Hasil Thresholding / Segmentasi", hasil["segmentasi"],
               "Thresholding mengubah peta 'kemerahan + kegelapan' menjadi "
               "hitam-putih. Segmentasi memisahkan area mencurigakan (putih) "
               "dari kulit normal (hitam).")
    with kol6:
        tampil("6. Hasil Deteksi Contour", hasil["kontur"],
               "Contour adalah garis tepi objek putih. Biru = semua contour, "
               "hijau = yang lolos filter (luas, bentuk, intensitas).",
               channels="BGR")

    tampil("7. Foto dengan Bounding Box Kandidat", hasil["deteksi"],
           "Bounding box (kotak merah) menunjukkan lokasi kandidat. Kotak "
           "oranye = area wajah yang dianalisis.", channels="BGR")

    # ---------------- 8-9. Ringkasan ----------------
    jumlah = len(hasil["kandidat"])
    tinggi, lebar = hasil["gambar_asli"].shape[:2]
    st.subheader("HASIL ANALISIS")
    st.code(
        f"Ukuran gambar           : {lebar} x {tinggi}\n"
        f"Mode                    : Grayscale\n"
        f"Jumlah kandidat jerawat : {jumlah}"
    )
    if jumlah > 0:
        st.error("TERDETEKSI KANDIDAT JERAWAT")
        st.write(f"Ditemukan **{jumlah}** kandidat area yang memiliki "
                 "karakteristik seperti jerawat.")
        st.dataframe(hasil["kandidat"], use_container_width=True)
    else:
        st.success("TIDAK ADA KANDIDAT")
        st.write("Tidak ditemukan kandidat jerawat berdasarkan parameter "
                 "pengolahan citra.")
    st.caption("Catatan: hasil merupakan analisis berdasarkan karakteristik "
               "citra dan bukan diagnosis medis.")