import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data(file_path):
    """
    Fungsi untuk memuat dataset, membersihkan data, melakukan encoding,
    membagi dataset, dan menstandarisasi fitur secara otomatis.
    """
    print("Memulai proses data preprocessing otomatis...")
    
    # 1. Memuat Dataset
    df = pd.read_csv(file_path)
    print(f"Data dimuat dengan dimensi awal: {df.shape}")
    
    # 2. Membersihkan Data Kosong & Duplikat
    df = df.drop_duplicates()
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
            
    # 3. Encoding Data Kategorikal
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    
    le = LabelEncoder()
    df['Month'] = le.fit_transform(df['Month'])
    
    df = pd.get_dummies(df, columns=['VisitorType'], drop_first=True)
    
    # 4. Membagi Data (Train-Test Split)
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Standarisasi Fitur Numerik (Scaling)
    numeric_cols = ['Administrative', 'Administrative_Duration', 'Informational', 
                    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
                    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
    
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print("Data preprocessing selesai! Data siap untuk dilatih.")
    return X_train, X_test, y_train, y_test

# Blok ini hanya akan dieksekusi jika file ini dijalankan langsung,
# sangat berguna untuk testing.

if __name__ == "__main__":
    # Menggunakan '../' karena dataset berada di luar folder 'preprocessing'
    PATH_DATA = '../online_shoppers_intention.csv'
    
    # Folder output akan otomatis dibuat di dalam folder 'preprocessing'
    OUTPUT_DIR = 'dataset_processed'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(PATH_DATA)
        
        # Menyimpan hasil 
        X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
        X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
        y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
        y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)
        
        print(f"File berhasil diproses dan disimpan di dalam folder 'preprocessing/{OUTPUT_DIR}/'")
    except FileNotFoundError:
        print(f"Error: File '{PATH_DATA}' tidak ditemukan. Pastikan dataset berada di root direktori.")