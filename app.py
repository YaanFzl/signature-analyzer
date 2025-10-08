import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from skimage import morphology, measure
import math
from io import BytesIO

st.set_page_config(page_title="Angle Distance Signature Analyzer", layout="wide")

st.title("🔍 Angle Distance Signature Analyzer")
st.markdown("Upload gambar untuk menganalisis objek menggunakan angle distance signature")


def create_figure(images, titles=None, figsize=(14, 6), cmap='gray'):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for i, (img, ax) in enumerate(zip(images, axes)):
        if img is None:
            ax.set_title("None")
            ax.axis('off')
            continue
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        if titles:
            ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    return fig


st.sidebar.header("⚙️ Parameter Preprocessing")

clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 3.0, 0.5)
median_kernel = st.sidebar.slider("Median Blur Kernel", 3, 9, 5, 2)
bilateral_d = st.sidebar.slider("Bilateral Filter d", 5, 15, 9, 2)
adapt_block = st.sidebar.slider("Adaptive Threshold Block Size", 11, 101, 51, 10)
adapt_c = st.sidebar.slider("Adaptive Threshold C", 2, 20, 10, 1)

st.sidebar.header("🔧 Morphology Parameters")
close_kernel_size = st.sidebar.slider("Closing Kernel Size", 5, 50, 25, 5)
open_kernel_size = st.sidebar.slider("Opening Kernel Size", 3, 15, 7, 2)
min_object_size = st.sidebar.slider("Min Object Size", 500, 5000, 2000, 500)
hole_area = st.sidebar.slider("Hole Fill Area", 500, 5000, 2000, 500)

# ---------- Upload gambar ----------
uploaded_file = st.file_uploader("Pilih gambar", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("❌ File tidak terbaca. Pastikan format gambar valid.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        st.success(f"✅ Gambar berhasil dimuat! Resolusi: {w} x {h}")

        # Tombol proses
        if st.button("🚀 Proses Gambar", type="primary"):
            with st.spinner("Memproses gambar..."):
                # ---------- Preprocessing ----------
                # 1) Convert to grayscale
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # 2) CLAHE
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)

                # 3) Denoise
                blur = cv2.medianBlur(gray_clahe, median_kernel)
                blur = cv2.bilateralFilter(blur, d=bilateral_d, sigmaColor=75, sigmaSpace=75)

                # 4) Adaptive threshold
                th_adapt = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, blockSize=adapt_block, C=adapt_c
                )

                # Fallback to Otsu
                if np.mean(th_adapt) < 1 or np.mean(th_adapt) > 250:
                    _, th_adapt = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 5) Morphology
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
                closed = cv2.morphologyEx(th_adapt, cv2.MORPH_CLOSE, kernel_close)

                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
                opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

                # 6) Remove small objects & fill holes
                bool_mask = opened.astype(bool)
                clean = morphology.remove_small_objects(bool_mask, min_size=min_object_size)
                clean = morphology.remove_small_holes(clean, area_threshold=hole_area)
                mask = (clean.astype(np.uint8) * 255)

                st.subheader("📊 Tahap Preprocessing")
                fig1 = create_figure(
                    [img_rgb, gray, gray_clahe, blur, th_adapt, opened, mask],
                    titles=["Original", "Gray", "CLAHE", "Blurred", "Adaptive Thresh", "Morphology", "Final Mask"]
                )
                st.pyplot(fig1)
                plt.close()

                labels = measure.label(mask, connectivity=2)
                props = measure.regionprops(labels)

                if len(props) == 0:
                    st.error("❌ Tidak ada objek terdeteksi. Coba ubah parameter.")
                else:
                    props_sorted = sorted(props, key=lambda p: p.area, reverse=True)
                    main = props_sorted[0]
                    main_mask = (labels == main.label).astype(np.uint8) * 255

                    # Contour
                    contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cnt = contours[0]

                    # Centroid
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        cy, cx = main.centroid
                        cx, cy = int(cx), int(cy)
                    else:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                    # Draw results
                    vis = img_rgb.copy()
                    cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(vis, (cx, cy), 6, (255, 0, 0), -1)

                    st.subheader("🎯 Objek Terdeteksi")
                    fig2 = create_figure([main_mask, vis], titles=["Object Mask", "Detected + Centroid"],
                                         figsize=(12, 6))
                    st.pyplot(fig2)
                    plt.close()

                    # ---------- Angle Distance Signature ----------
                    max_radius = int(math.hypot(w, h))
                    mask_bool = (main_mask > 0)
                    signature = np.zeros(360, dtype=np.float32)

                    progress_bar = st.progress(0)
                    for theta in range(360):
                        rad = math.radians(theta)
                        last_inside = 0
                        found_inside = False

                        for r in range(0, max_radius, 1):
                            x = int(round(cx + r * math.cos(rad)))
                            y = int(round(cy + r * math.sin(rad)))

                            if x < 0 or x >= w or y < 0 or y >= h:
                                break

                            if mask_bool[y, x]:
                                last_inside = r
                                found_inside = True
                            else:
                                if found_inside:
                                    break

                        signature[theta] = last_inside
                        if theta % 10 == 0:
                            progress_bar.progress((theta + 1) / 360)

                    progress_bar.empty()

                    if np.max(signature) == 0:
                        st.error("❌ Signature kosong. Centroid mungkin tidak di dalam objek.")
                    else:
                        # Plot raw signature
                        st.subheader("📈 Raw Angle Distance Signature")
                        fig3, ax3 = plt.subplots(figsize=(12, 4))
                        ax3.plot(np.arange(360), signature)
                        ax3.set_title("Raw Angle Distance Signature (r vs theta)")
                        ax3.set_xlabel("Sudut (derajat)")
                        ax3.set_ylabel("Jarak (pixel)")
                        ax3.grid(True)
                        st.pyplot(fig3)
                        plt.close()

                        # Normalization
                        r_min = np.min(signature)
                        r_max = np.max(signature)

                        if r_max - r_min == 0:
                            r_norm = np.zeros_like(signature)
                        else:
                            r_norm = (signature - r_min) / (r_max - r_min)

                        # Rotation alignment
                        start_idx = int(np.argmin(r_norm))
                        r_norm_aligned = np.roll(r_norm, -start_idx)

                        st.subheader("📊 Normalized & Aligned Signature")
                        fig4, ax4 = plt.subplots(figsize=(12, 4))
                        ax4.plot(np.arange(360), r_norm_aligned)
                        ax4.set_title(f"Normalized & Aligned Signature (start at theta={start_idx})")
                        ax4.set_xlabel("Sudut (derajat, rotated)")
                        ax4.set_ylabel("Normalized distance")
                        ax4.grid(True)
                        st.pyplot(fig4)
                        plt.close()

                        # Polar plot
                        st.subheader("🎯 Polar Plot")
                        angles = np.deg2rad(np.arange(360))
                        fig5 = plt.figure(figsize=(6, 6))
                        ax5 = fig5.add_subplot(111, polar=True)
                        ax5.plot(angles, r_norm_aligned)
                        ax5.set_title("Polar Plot Normalized Signature")
                        st.pyplot(fig5)
                        plt.close()

                        # Download signature data
                        st.subheader("💾 Download Data")
                        col1, col2 = st.columns(2)

                        with col1:
                            # Raw signature
                            raw_data = "\n".join([f"{i},{signature[i]}" for i in range(360)])
                            st.download_button(
                                label="📥 Download Raw Signature (CSV)",
                                data=f"angle,distance\n{raw_data}",
                                file_name="raw_signature.csv",
                                mime="text/csv"
                            )

                        with col2:
                            # Normalized signature
                            norm_data = "\n".join([f"{i},{r_norm_aligned[i]}" for i in range(360)])
                            st.download_button(
                                label="📥 Download Normalized Signature (CSV)",
                                data=f"angle,normalized_distance\n{norm_data}",
                                file_name="normalized_signature.csv",
                                mime="text/csv"
                            )

                        st.success("✅ Proses selesai!")

                        with st.expander("💡 Tips Optimasi"):
                            st.markdown("""
                            **Jika objek masih terfragmentasi/noisy:**
                            - Naikkan **Closing Kernel Size** (mis. 35-45) untuk menggabungkan bagian terpisah
                            - Turunkan **Min Object Size** jika objek kecil hilang
                            - Atur **Adaptive Threshold Block Size** (nilai ganjil, 31-101)
                            - Sesuaikan **CLAHE Clip Limit** untuk kontras yang lebih baik
                            """)

else:
    st.info("👆 Silakan upload gambar untuk memulai analisis")

    with st.expander("ℹ️ Tentang Aplikasi"):
        st.markdown("""
        Aplikasi ini menganalisis bentuk objek dalam gambar menggunakan **Angle Distance Signature**.

        **Fitur:**
        - Preprocessing otomatis (CLAHE, denoising, thresholding)
        - Deteksi kontur dan centroid objek
        - Angle distance signature (360 derajat)
        - Normalisasi dan rotational alignment
        - Visualisasi polar plot
        - Export data ke CSV

        **Cara Penggunaan:**
        1. Upload gambar objek
        2. Sesuaikan parameter di sidebar (opsional)
        3. Klik tombol "Proses Gambar"
        4. Lihat hasil analisis dan download data
        """)