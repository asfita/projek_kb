import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Data simulasi restoran dan statistik mereka
data = {
    'Restoran': ['Resto A', 'Resto B', 'Resto C', 'Resto D', 'Resto E', 'Resto F', 'Resto G', 'Resto H'],
    'Jumlah_Pengunjung': [200, 150, 300, 250, 100, 180, 90, 220],
    'Rating': [4.5, 3.8, 4.9, 4.2, 3.5, 4.0, 3.2, 4.6],
    'Keberagaman_Menu': [50, 30, 60, 45, 20, 35, 15, 55],
    'Harga_Rata_Rata': [80, 70, 100, 90, 60, 75, 50, 85],  # Harga dalam ribuan
    'Populer': [1, 0, 1, 1, 0, 0, 0, 1]  # 1 jika restoran populer, 0 jika tidak
}

# Membuat DataFrame dari data
df = pd.DataFrame(data)

# Memisahkan fitur dan label
X = df[['Jumlah_Pengunjung', 'Rating', 'Keberagaman_Menu', 'Harga_Rata_Rata']]  # Fitur
y = df['Populer']  # Label: Populer (1 = ya, 0 = tidak)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data menggunakan StandardScaler
scaler = StandardScaler()

# Menormalkan data latih dan uji
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Inisialisasi model K-NN dengan jumlah tetangga terdekat k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Melatih model dengan data latih
knn.fit(X_train_scaled, y_train)

# Memprediksi hasil pada data uji
y_pred = knn.predict(X_test_scaled)

# Evaluasi model
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred, zero_division=1))

# Menampilkan restoran yang populer (Populer = 1)
popular_restaurants = df[df['Populer'] == 1]
print("\nRestoran Populer:")
for index, resto in popular_restaurants.iterrows():
    print(f"{resto['Restoran']} - Rating: {resto['Rating']}, Pengunjung: {resto['Jumlah_Pengunjung']}")

# Menentukan restoran dengan jumlah pengunjung terbanyak
top_restaurant = df[df['Jumlah_Pengunjung'] == df['Jumlah_Pengunjung'].max()]

print("\nRestoran dengan Pengunjung Terbanyak:")
for index, resto in top_restaurant.iterrows():
    print(f"{resto['Restoran']} - Pengunjung: {resto['Jumlah_Pengunjung']}")
