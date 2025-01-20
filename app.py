import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd  # Importing pandas
from sklearn.linear_model import LogisticRegression

# Memuat dataset
@st.cache_data
def load_data():
    data = pd.read_csv('diabetes.csv')
    return data

# Fungsi utama untuk menjalankan aplikasi
def main():
    
    # Memuat data
    data = load_data()
    df = data.copy()
    x = df.drop("Outcome", axis=1)  # Fitur
    y = df["Outcome"]  # Variabel target

    # Model regresi logistik untuk prediksi
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(x, y)  # Melatih model pada dataset

# Set up the title of the app
st.title("Aplikasi Prediksi Diabetes")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.header("Masukan Data")
    input_data = (
        st.number_input("Jumlah Kehamilan", min_value=0),
        st.number_input("Glukosa", min_value=0),
        st.number_input("Tekanan Darah", min_value=0),
        st.number_input("Skin Thickness", min_value=0),
    )

with col2:
    st.header("More Input Data")
    input_data += (
        st.number_input("Insulin", min_value=0),
        st.number_input("BMI", min_value=0.0),
        st.number_input("Fungsi Pedigree Diabetes", min_value=0.0),
        st.number_input("Usia", min_value=0),
    )

if st.button("Prediksi"):
        input_data_as_array = np.array(input_data)
        input_data_reshape = input_data_as_array.reshape(1, -1)

        # Logika prediksi tiruan
        def mock_predict(input_data):
            glucose = input_data[1]  # Glukosa adalah input kedua
            if glucose > 140:  # Ambang batas contoh untuk prediksi diabetes
                return [1]  # Prediksi diabetes
            else:
                return [0]  # Tidak ada diabetes

        prediction = mock_predict(input_data)  # Menggunakan fungsi prediksi tiruan

        # Menyimpan prediksi ke file .sav
        file_name = "predictions.sav"
        try:
            # Memuat prediksi yang ada jika tersedia
            predictions = joblib.load(file_name)
        except FileNotFoundError:
            # Menginisialisasi daftar prediksi jika file tidak ditemukan
            predictions = []

        # Menambahkan prediksi baru
        predictions.append({
            "input_data": input_data,
            "prediction": "Pasien Terkena Diabetes" if prediction[0] == 1 else "Pasien Tidak Terkena Diabetes"
        })

        # Menyimpan prediksi yang diperbarui
        joblib.dump(predictions, file_name)

        if prediction[0] == 0:
            st.success('Pasien Tidak Terkena Diabetes')
        else:
            st.error('Pasien Terkena Diabetes')

# Create three columns for visualizations
col3, col4, col5 = st.columns(3)

with col3:
    st.header("Visualisasi 1")
    # Visualizations
    # Plotting the glucose distribution using seaborn
    plt.figure(figsize=(8, 6))
    sns.histplot(data=['Glucose'], kde=True, color='blue', bins=20)
    plt.title('Distribusi Glukosa pada Pasien Diabetes')
    sns.boxplot(data=np.random.normal(size=(5, 4)))
    plt.xlabel('Glukosa')
    plt.ylabel('Frekuensi')
    st.pyplot(plt)

with col4:
    st.header("Visualisasi 2")
    # Placeholder for visualization 2
    st.write("Scatter Plot")
    # Example visualization
    plt.figure(figsize=(5, 3))
    plt.scatter(np.random.rand(100), np.random.rand(100))
    st.pyplot(plt)

with col5:
    st.header("Visualisasi 3")
    # Placeholder for visualization 3
    st.write("Box Plot")
    # Example visualization
    plt.figure(figsize=(5, 3))
    sns.boxplot(data=np.random.normal(size=(10, 4)))
    st.pyplot(plt)

# Call the main function to run the app
if __name__ == "__main__":
    main()
