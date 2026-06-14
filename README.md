# 🔌 Klasifikasi Kondisi Stopkontak Listrik: ML vs Deep Learning

Perbandingan performa metode **Machine Learning** dan **Deep Learning** dalam klasifikasi kondisi stopkontak listrik berbasis citra digital.

---

## 📋 Daftar Isi

- [Deskripsi Proyek](#deskripsi-proyek)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
- [Hasil](#hasil)
- [Explainable AI (XAI)](#explainable-ai-xai)
- [Tools & Framework](#tools--framework)
- [Struktur Proyek](#struktur-proyek)
- [Kesimpulan](#kesimpulan)
- [Author](#author)
- [Referensi](#referensi)

---

## 📌 Deskripsi Proyek

Proyek ini membandingkan performa beberapa metode klasifikasi citra dalam mendeteksi kondisi stopkontak listrik. Terdapat **tiga kelas** yang diklasifikasikan:

| Kelas | Deskripsi |
|-------|-----------|
| 🟢 Kosong | Stopkontak dalam kondisi kosong (tidak ada steker) |
| 🟡 Terisi | Stopkontak terisi sebagian |
| 🔴 Penuh | Stopkontak terisi penuh |

### Model yang Dibandingkan

**Machine Learning**
- SVM + HOG
- EfficientNetB0 + SVM

**Deep Learning**
- EfficientNetB0
- EfficientNetV2B0
- YOLOv5s
- YOLOv11s

---

## 📦 Dataset

Dataset dikumpulkan secara mandiri berupa citra kondisi stopkontak listrik.

| Keterangan | Detail |
|------------|--------|
| Data awal | 510 gambar |
| Setelah augmentasi | 7.500 gambar |
| Distribusi per kelas | 2.500 gambar |

### Teknik Augmentasi Data

Menggunakan library **Albumentations** dengan teknik berikut:

- Horizontal & Vertical Flip
- Rotation, Translation, Scaling
- Brightness & Contrast
- Hue Saturation Value (HSV)
- Gaussian Blur & Gaussian Noise
- Random Shadow
- Gamma Correction

---

## 🧪 Metodologi

### 1. SVM + HOG
- Ekstraksi fitur menggunakan **Histogram of Oriented Gradients (HOG)**
- Klasifikasi menggunakan **Support Vector Machine (SVM)**
- Hyperparameter tuning dengan **GridSearchCV**
- Evaluasi menggunakan **Stratified 5-Fold Cross Validation**

### 2. EfficientNetB0 + SVM
- **EfficientNetB0** sebagai feature extractor (Transfer Learning dari bobot ImageNet)
- **SVM** sebagai classifier

### 3. EfficientNetB0
- Transfer Learning dengan fine-tuning pada classifier layer
- Interpretabilitas menggunakan **Grad-CAM**

### 4. EfficientNetV2B0
- Transfer Learning dengan Global Average Pooling
- Interpretabilitas menggunakan **Grad-CAM**

### 5. YOLOv5s
- Transfer Learning
- Evaluasi: Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Interpretabilitas menggunakan **LIME**

### 6. YOLOv11s
- 5-Fold Cross Validation
- Evaluasi: Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Interpretabilitas menggunakan **LIME**

---

## 📊 Hasil

| Model | Kategori | Accuracy / mAP@0.5 |
|-------|----------|-------------------|
| SVM + HOG | Machine Learning | 81.37% |
| EfficientNetB0 + SVM | Machine Learning | 97.73% |
| EfficientNetB0 | Deep Learning | 97.33% |
| EfficientNetV2B0 | Deep Learning | 95.64% |
| YOLOv5s | Deep Learning | **99.40%** ⭐ |
| YOLOv11s | Deep Learning | 99.03% |

### 🏆 Model Terbaik: YOLOv5s

| Metrik | Nilai |
|--------|-------|
| Precision | 97.9% |
| Recall | 98.7% |
| mAP@0.5 | **99.4%** |
| mAP@0.5:0.95 | 96.2% |

---

## 🔍 Explainable AI (XAI)

Penelitian menerapkan metode XAI untuk meningkatkan interpretabilitas model:

| Metode | Digunakan pada |
|--------|---------------|
| **Feature Importance** | EfficientNetB0 + SVM |
| **Grad-CAM** | EfficientNetB0, EfficientNetV2B0 |
| **LIME** | YOLOv5s, YOLOv11s |

> Hasil visualisasi menunjukkan bahwa model berfokus pada area **stopkontak, steker, adaptor, dan kabel** saat melakukan klasifikasi.

---

## 🛠️ Tools & Framework

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

| Tool | Kegunaan |
|------|----------|
| Python | Bahasa pemrograman utama |
| Scikit-Learn | SVM, GridSearchCV, evaluasi |
| TensorFlow / Keras | EfficientNet models |
| Ultralytics YOLO | YOLOv5s & YOLOv11s |
| OpenCV | Pemrosesan citra |
| Albumentations | Augmentasi data |
| Roboflow | Manajemen dataset |
| Google Colab | Lingkungan pelatihan model |
| Streamlit | Aplikasi demo interaktif |

---

## 📁 Struktur Proyek

```
📦 klasifikasi-stopkontak/
├── 📂 dataset/              # Dataset citra stopkontak
├── 📂 svm_hog/              # Implementasi SVM + HOG
├── 📂 efficientnetb0_svm/   # Implementasi EfficientNetB0 + SVM
├── 📂 efficientnetb0/       # Implementasi EfficientNetB0
├── 📂 efficientnetv2b0/     # Implementasi EfficientNetV2B0
├── 📂 yolov5s/              # Implementasi YOLOv5s
├── 📂 yolov11s/             # Implementasi YOLOv11s
├── 📂 streamlit_app/        # Aplikasi Streamlit
├── 📂 models/               # Model tersimpan
├── 📂 notebooks/            # Jupyter Notebooks
└── 📄 README.md
```

---

## 📝 Kesimpulan

- Model **Deep Learning** secara konsisten mengungguli metode **Machine Learning konvensional** pada tugas klasifikasi kondisi stopkontak listrik.
- **YOLOv5s** meraih performa terbaik secara keseluruhan dengan **mAP@0.5 sebesar 99.4%**.
- **EfficientNetB0 + SVM** menjadi model klasifikasi terbaik di kategori hybrid dengan **akurasi 97.73%**.
- Transfer Learning terbukti efektif meski dataset relatif kecil sebelum augmentasi.

---

## 👤 Author

**Ghani Mudzakir**
- NIM: `2310817110011`
- Program Studi: Teknologi Informasi
- Fakultas: Teknik
- Universitas: Lambung Mangkurat

---

## 🔗 Link

| Sumber | URL |
|--------|-----|
| 📁 GitHub Repository | [Isi link repository] |
| 🗃️ Dataset Roboflow | [Isi link dataset] |
| 🖥️ Demo Streamlit | [Isi link aplikasi] |

---

## 📚 Referensi

- Basthikodi et al. (2024)
- Celik & Inik (2024)
- Sabir & Mehmood (2024)
- Giraldo-Roldán et al. (2026)
- Lee et al. (2024)
- Aftab et al. (2026)

---

<p align="center">
  Made with ❤️ by Ghani Mudzakir · Universitas Lambung Mangkurat
</p>
