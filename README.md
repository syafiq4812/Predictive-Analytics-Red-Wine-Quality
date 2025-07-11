# Predictive-Analytics-Red-Wine-Quality

## Domain proyek

Domain yang dipilih untuk proyek *machine learning* ini adalah Industri, dengan judul **Predictive Analytics: Kualitas Anggur Merah**

### Latar Belakang

![foto Anggur Merah](https://i.ibb.co/zTTk1srw/winee.jpg)

Industri anggur merah yang marak di distribusikan di indonesia maupun di luar negeri, namun penilaian kualitas anggur merah dilakukan dengan menggunakan metode manual seperti uji organoleptik oleh sommelier, yang rentan terhadap subjektivitas dan ketidakkonsistenan.Untuk mengatasi hal tersebut, proyek ini menerapkan machine learning predictive analysis untuk memprediksi kualitas anggur merah secara objektif berdasarkan parameter kimia seperti kadar alkohol, pH, asam volatil, dan sulfat, menggunakan dataset dari UCI Machine Learning Repository [(Cortez et al., 2009)](https://archive.ics.uci.edu/dataset/186/wine+quality).Pendekatan ini tidak hanya meningkatkan akurasi dan efisiensi penilaian, tetapi juga membantu produsen dalam mengoptimalkan proses produksi dan menjaga standar kualitas berbasis data.

## Business Understanding

Pengembangan model prediksi kualitas Anggur merah berbasis machine learning menawarkan manfaat dari berbagai pihak dalam industri wine, seperti produsen anggur, distributor, konsumen, dan pihak regulator. Model ini dapat membantu meningkatkan kualitas anggur merah dengan mengoptimalkan proses produksi dengan mengidentifikasi faktor kimia yang paling berpengaruh terhadap kualitas. Contoh potensi manfaat hasil prediksi kualitas anggur merah yang akurat dapat membantu regulator untuk menggunakan model sebagai alat standarisasi kualitas, memastikan fair competition di pasar. Dengan demikian, solusi ini tidak hanya meningkatkan efisiensi operasional tetapi juga menciptakan nilai tambah bagi seluruh rantai pasok (supply chain) industri anggur.

### Problem Statements

Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:

- Bagaimana membuat model machine learning yang dapat memprediksi kualitas anggur merah berdasarkan data visual dan sensorik?
- Model yang seperti apa yang memiliki akurasi paling baik?
- Bagaimana model ini dapat membantu produsen dan distributor dalam meningkatkan kualitas dan nilai jual apel?

### Gols

Tujuan dari proyek ini adalah:

- Membuat model machine learning yang dapat memprediksi kualitas Anggur merah berdasarkan data visual dan sensorik.
- Membandingkan beberapa algoritma model untuk menemukan akurasi terbaik dalam memprediksi kualitas Anggur merah.
- Mengembangkan aplikasi yang mudah digunakan untuk membantu produsen dan distributor dalam menggunakan model machine learning untuk memprediksi kualitas anggur merah.

### solution statement

- Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui kolerasi matrix antar fitur dan mendeteksi outlier.
- Melakukan proses data cleaning dan normalisai data agar mendapat prediksi yang baik.
- Membuat beberapa variasi model untuk mendapatkan model yang paling baik dari beberapa model yang telah dibuat untuk prediksi kualitas apel. Diantaranya adalah menggunakan:
 - K-Nearest Neighbor (KNN) adalah Algoritma ini bekerja dengan menemukan "k" titik data (tetangga) terdekat dengan input yang diberikan dan membuat prediksi berdasarkan kelas mayoritas (untuk klasifikasi) atau nilai rata-rata (untuk regresi).[(1)](https://www.geeksforgeeks.org/k-nearest-neighbours/)
 - Random Forest adalah algoritma machine learning yang kuat yang dapat digunakan untuk berbagai tugas termasuk regresi dan klasifikasi.Dalam tugas klasifikasi, _Random Forest Classifier_ memprediksi hasil kategoris berdasarkan data masukan. Klasifikasi ini menggunakan beberapa pohon keputusan dan menghasilkan label yang memiliki suara terbanyak di antara semua prediksi pohon individual .[(2)](https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/)
 - Gradient Boosting Classifier adalah Algoritme ini membangun model aditif secara bertahap; memungkinkan pengoptimalan fungsi kerugian yang dapat dibedakan secara acak. Pada setiap tahap, n_classes_pohon regresi disesuaikan dengan gradien negatif fungsi kerugian, misalnya kerugian log biner atau multikelas. Klasifikasi biner adalah kasus khusus di mana hanya satu pohon regresi yang diinduksi.[(3)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

## Data Understanding

### EDA-Deskripsi Variable

**Informasi Data**

| Jenis       | Keterangan                                                                 |
|-------------|---------------------------------------------------------------------------|
| Title       | Red wine Quality                                                           |
| Source      | [Kaggle ](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?select=winequality-red.csv)                                                                   |
| Maintainer  | [UCI Machine Learning](https://www.kaggle.com/organizations/uciml)                                                 |
| License     | Other (specified in description)                                          |
| Visibility  | Publik                                                                    |
| Tags        | Earth And Nature, Education, Beginner,Alcohol, Benchamark |
| Usability   | 8.82                                                                     |

Berikut Informasi pada dataset : Data yang digunakan dalam pembuatan model merupakan data primer, data ini didapat dari sebuah University Of California, yang disediakan secara publik di kaggle dengan nama dataset Red Wine Quality

| fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free sulfur dioxide | total sulfur dioxide | density | pH  | sulphates | alcohol | quality |
|--------------|------------------|-------------|----------------|-----------|---------------------|----------------------|---------|-----|-----------|---------|---------|
| 7.4          | 0.70             | 0.00        | 1.9            | 0.076     | 11.0                | 34.0                 | 0.9978  | 3.51 | 0.56      | 9.4     | 5       |
| 7.8          | 0.88             | 0.00        | 2.6            | 0.098     | 25.0                | 67.0                 | 0.9968  | 3.20 | 0.68      | 9.8     | 5       |
| 7.8          | 0.76             | 0.04        | 2.3            | 0.092     | 15.0                | 54.0                 | 0.9970  | 3.26 | 0.65      | 9.8     | 5       |
| 11.2         | 0.28             | 0.56        | 1.9            | 0.075     | 17.0                | 60.0                 | 0.9980  | 3.16 | 0.58      | 9.8     | 6       |
| 7.4          | 0.70             | 0.00        | 1.9            | 0.076     | 11.0                | 34.0                 | 0.9978  | 3.51 | 0.56      | 9.4     | 5       |
| 7.4          | 0.66             | 0.00        | 1.8            | 0.075     | 13.0                | 40.0                 | 0.9978  | 3.51 | 0.56      | 9.4     | 5       ||

Tabel 1. EDA Deskripsi Variabel

Dilihat dari _Tabel 1. EDA Deskripsi Variabel_ dataset ini telah di *bersihkan* dan *normalisasi* terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula.
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 1599 sample dengan 12 fitur.
- Dataset memiliki 11 fitur bertipe float64 dan 1 fitur bertipe integer.
- Terdapat 240 jumlah duplikasi data.

### Variable - variable pada dataset

- `fixed acidity`; Asam non-volatil yang memengaruhi kestabilan dan rasa.
- `volatile acidity`;	Asam volatil yang tinggi bisa menyebabkan rasa tidak enak.
- `citric acid`; Asam sitrat, penambah kesegaran dan kompleksitas rasa.
- `presidual sugar`; Gula tersisa setelah fermentasi, memengaruhi tingkat kemanisan.
- `chlorides`; Kadar garam, memengaruhi rasa asin/mineral.
- `free sulfur dioxide`; SO₂ bebas, berfungsi sebagai pengawet dan antioksidan.
- `total sulfur dioxide`; Total SO₂ (bebas + terikat), terkait dengan preservasi dan aroma.
- `density`;	Massa jenis wine, berkorelasi dengan alkohol/gula.
-` pH`; Tingkat keasaman (skala logaritmik). pH rendah = lebih asam.
- `sulphates`; Kadar sulfat (e.g., potassium sulphate) yang memengaruhi SO₂ dan rasa.
- `alcohol`; Persentase alkohol, berpengaruh pada body dan rasa.
- `quality`; Target/label kualitas wine (biasanya skala 0-10).

### EDA - Univariate Analysis

Analisis Univariat (Data Kategori)

![Univariate Analysis](https://i.ibb.co/cj8pN8C/data-univariet.png)

gambar 1a. Analisis Univariat (Data Numerik)

Berdasarkan data gambari diatas memiliki karakteristik, yaitu:
  - Dilihat dari distribusi data numerik _fixed acidity_, yang mempengaruhi Kestabilan dan rasa berkisar dari 7 sampai 8, dan memiliki nilai rata-rata _Mean_ adalah 8.31.
  -Untuk menambah kesegaran dan kompleksitas rasa berkisar dari 0.0 sampai 0.1, dan memiliki nilai rata-rata _Mean_ adalah 0.27.
  - Rata-rata tingkat kemanasisan dari hasil fermentasi 2.54 dan _Max_ tingkat kemanisan 1.00.
  - Rata-rata tingkat Keasaman berada di 3.31, dan _max_ berada di 4.01.
  - Masa jenis wine yang berkorelasi dengan alkohol/gula berkisar dari 0.996 sampai 0.998.
  - persentase alcohol rata-rata di 10.42, dengan nilai _Max_ 14.90.

### EDA - Multivariate Analysis

![Multivariate Analysis](https://i.ibb.co/DHNKRKnD/Multivarieta1.png)


Gambar 2a. Analisis Multivariat

![Multivariate Analysis](https://i.ibb.co/jv0QkcKR/Multivariete2.png)


Gambar 2b. Analisis Matriks Korelasi

Pada _Gambar 2a. Analisis Multivariat_, dengan menggunakan fungsi _pairplot_ dari _library seaborn_, tampak terlihat relasi pasangan dalam dataset menunjukan pola acak. Pada pola sebaran data grafik pairplot, terterlihat bahwa _Alcohol_ dan _quality_ memiliki korelasi positif yang cukup kuat dengan kualitas menunjukkan bahwa wine dengan kadar alkohol lebih tinggi cenderung mendapat penilaian lebih baik.

Pada _Gambar 2b. Analisis Matriks Korelasi_, merupakan _Correlation Matrix_ menunjukkan hubungan antar fitur dalam nilai korelasi. Jika diamati, fitur _fixed Acidity_ memiliki skor korelasi yang cukup besar `0.66` dengan fitur target _Citric acid_ .

## Data Preparation

Pada proses _Data Preparation_ dilakukan kegiatan seperti _Data Gathering_, _Data Assessing_, dan _Data Cleaning_. Pada proses Data Gathering, data diimpor sedemikian rupa agar bisa dibaca dengan baik menggunakan dataframe Pandas. Untuk proses Data Assessing, berikut adalah beberapa pengecekan yang dilakukan:
- Duplicate data (data yang serupa dengan data lainnya).
- Missing value (data atau informasi yang "hilang" atau tidak tersedia)
- Outlier (data yang menyimpang dari rata-rata sekumpulan data yang ada).

Pada proses _Data Cleaning_ yang dilakukan adalah seperti:
- Converting Column Type (Mengubah tipe suatu kolom).
- Train Test Split (membagi data menjadi data latih dan data uji).
- Normalization (mentransformasi data ke dalam skala yang seragam sehingga semua fitur atau atribut memiliki rentang nilai yang sebanding).

| id_wine | fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | free sulfur dioxide | total sulfur dioxide | density  | pH  | sulphates | alcohol | quality |
|---------|--------------|------------------|-------------|----------------|-----------|---------------------|----------------------|----------|-----|-----------|---------|---------|
| 142     | 5.2          | 0.340            | 0.00        | 1.8            | 0.050     | 27.0                | 63.0                 | 0.99160  | 3.68 | 0.79      | 14.0    | 6       |
| 144     | 5.2          | 0.340            | 0.00        | 1.8            | 0.050     | 27.0                | 63.0                 | 0.99160  | 3.68 | 0.79      | 14.0    | 6       |
| 131     | 5.6          | 0.500            | 0.09        | 2.3            | 0.049     | 17.0                | 99.0                 | 0.99370  | 3.63 | 0.63      | 13.0    | 5       |
| 132     | 5.6          | 0.500            | 0.09        | 2.3            | 0.049     | 17.0                | 99.0                 | 0.99370  | 3.63 | 0.63      | 13.0    | 5       |
| 1488    | 5.6          | 0.540            | 0.04        | 1.7            | 0.049     | 5.0                 | 13.0                 | 0.99420  | 3.72 | 0.58      | 11.4    | 5       |
| ...     | ...          | ...              | ...         | ...            | ...       | ...                 | ...                  | ...      | ... | ...       | ...     | ...     |
| 391     | 13.7         | 0.415            | 0.88        | 2.9            | 0.085     | 17.0                | 43.0                 | 1.00140  | 3.06 | 0.80      | 10.0    | 6       |
| 243     | 15.0         | 0.210            | 0.44        | 2.2            | 0.075     | 10.0                | 24.0                 | 1.00005  | 3.07 | 0.84      | 9.2     | 7       |
| 244     | 15.0         | 0.210            | 0.44        | 2.2            | 0.075     | 10.0                | 24.0                 | 1.00005  | 3.07 | 0.84      | 9.2     | 7       |
| 554     | 15.5         | 0.645            | 0.49        | 4.2            | 0.095     | 10.0                | 23.0                 | 1.00315  | 2.82 | 0.74      | 11.1    | 5       |
| 555     | 15.5         | 0.645            | 0.49        | 4.2            | 0.095     | 10.0                | 23.0                 | 1.00315  | 2.82 | 0.74      | 11.1    | 5       |

Tabel 2. Melihat Data Duplikasi

Pada proyek ini tidak ditemukan missing value namun ditemukan data duplikasi sebanyak 240 data terlampir. Adapun metode yang dilakukan untuk mengatasi hal ini adalah dengan menerapkan Dropping yaitu menghapus data duplikasi yang teridentivikasi. Adapun untuk outlier juga dilakukan dengan metode _dropping_ menggunakan metode IQR. IQR dihitung dengan mengurangkan kuartil ketiga (Q3) dari kuartil pertama (Q1) sebagaimana rumus berikut.

$$IQR = Q_3 - Q_1$$

- Q1 adalah kuartil pertama
- Q3 adalah kuartil ketiga.

Setelah menggunakan metode IQR untuk menghilangkan _outlier_ pada dataset jumlah dataset menjadi `1005` yang awalnya adalah `1359`.
Pada proyek ini digunakan _Train Test Split_ pada library  *sklearn.model_selection* untuk membagi dataset menjadi data latih dan data uji dengan pembagian sebesar 20:80 dan random state sebesar 42. Pada proyek kasus ini digunakan _Normalization_ pada library _sklearn.preprocessing.MinMaxScaler_ untuk menormalisasi dataset. Semua proses ini diperlukan dalam rangka membuat model yang baik.

## Modeling
Algoritma pada proyek ini melakukan pemodelan dengan 3 algoritma, yaitu:

_K-Nearest Neighbors (KNN)_ adalah algoritma machine learning yang sederhana dan mudah dipahami untuk klasifikasi dan regresi. Algoritma ini bekerja dengan menemukan k tetangga terdekat dari data baru dan kemudian menggunakan kategori atau nilai rata-rata dari tetangga tersebut untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
-  `n_neighbors` jumlah tetangga terdekat.
- `weight = distance` Tetangga yang lebih dekat memiliki pengaruh lebih besar.

Keunggulan _KNN_ :
- Dapat digunakan untuk klasifikasi dan regresi.
- Sederhana dan mudah dipahami.

Kerugian _KNN_ :
- Sensitif terhadap outlier.
- Membutuhkan banyak memori dan waktu komputasi untuk dataset besar.
- Sulit untuk memilih nilai K yang optimal.

_Random Forest_ adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
- `max_depth` kedalaman maksimum.
- `n_estimators` Jumlah pohon keputusan yang akan dibuat dalam ensemble.

Keunggulan _Random Forest_ :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap outlier.

Kerugian _Random Forest_ :
- Cenderung overfit pada dataset kecil.
- Membutuhkan banyak waktu komputasi untuk pelatihan.
- Sulit untuk diinterpretasikan.

_Gradient Boosting Classifier (GBC)_ adalah model ensemble learning yang membangun beberapa weak learners (biasanya pohon keputusan kecil) secara bertahap, di mana setiap model baru mencoba memperbaiki kesalahan model sebelumnya. Adapun parameter yang digunakan pada proyek ini adalah:
- `n_estimators` Jumlah pohon keputusan yang akan dibuat dalam ensemble.
- `max_depth` kedalaman maksimum.

Keunggulan _GBC_ :
- Memiliki akurasi prediksi yang tinggi.
- Dapat mengabaikan fitur yang didak penting.
- cocok untuk hubungan kompleks dalam data.

Kerugian _GBC_ :
- Cenderung training yang lambat, karena bekerja secara berurutan (tidak seperti Random forest).
- Sensitif terhadap Outlier, karena berfokus pada residual (error).
- Rentang Overfitting terlalu banyak/besar dan tidak sesuai parameter yang digunakan.

Parameter yang digunakan adalah:
- `n_estimators` Jumlah pohon keputusan yang akan dibuat dalam ensemble.
- `random_stat`  pengambilan sampel secara acak.
- `max_depth` Kedalaman maksimum pohon keputusan individual.
- `n_jobs` mempercepat pelatihan pada sistem dengan beberapa core CPU.

## Evaluation

Dalam tahap evaluasi, metrik yang digunakan adalah `accuracy`
Accuracy didapatkan dengan menghitung persentase dari jumlah prediksi yang benar dibagi dengan jumlah seluruh prediksi. Rumus:

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%$$

*Penjelasan*
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

Rumus ini memecah akurasi menjadi rasio antara data yang diklasifikasikan dengan benar (TP dan TN) dengan jumlah total data. Mengalikan dengan 100% mengubah rasio menjadi persentase.

Berikut hasil accuracy 3 buah model yang latih:

| Model   | Accuracy |
| --------- | ---------- |
| KNN | 0.91 |
| RandomForest  | 0.92 |
| Gradient Boosting Classifier  | 0.91 |

Tabel 3. Hasil Accuracy

![Plot Accuracy](https://i.ibb.co/v4gMjMTf/Model-Acuracy.png)

Gambar 3. Visualisasi Accuracy Model

Dilihat dari _Tabel 3. Hasil Accuracy_ dan _Gambar 3. Visualisasi Accuracy Model_ tersebut dapat diketahui bahwa model dengan algoritma _Rendom Forest Classifier _ memiliki Accuracy yang lebih tinggi dengan accuracy `92%` . Untuk itu model tersebut yang akan dipilih untuk digunakan. Diharapkan dengan model yang telah dikembangan dapat memprediksi kualitas Anggur merah dengan baik menggunakan _Rendom Forest Classifier_. Alasan mengapa metode _Rendom Forest Classifier_ yang dipilih karena algoritma machine learning yang kuat, dan cocok untuk diguanakan untuk berbagai tugas termasuk regresi dan klasifikasi.Dalam tugas klasifikasi, _Random Forest Classifier_ memprediksi hasil kategoris berdasarkan data masukan. Klasifikasi ini menggunakan beberapa pohon keputusan dan menghasilkan label yang memiliki suara terbanyak di antara semua prediksi pohon individual.

## Referensi
1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/186/wine+quality.

2. GeeksforGeeks Team. (2023). Understanding K-Nearest Neighbors Algorithm. Diakses dari https://www.geeksforgeeks.org/k-nearest-neighbours/

3. GeeksforGeeks Team. (2023). Implementing Random Forest Classifier in Python with Scikit-learn. Diakses dari https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/

4. Pedregosa, F., et al. (2023). Gradient Boosting Classifier Implementation in scikit-learn. scikit-learn Documentation. Diakses dari https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
