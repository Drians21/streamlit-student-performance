import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Membaca dataset
df = pd.read_csv('StudentsPerformance.csv')

#rename columns
df.columns = df.columns.str.replace(" ", "_").copy()
df = df.rename({'parental_level_of_education':'parent_education', 
                        'race/ethnicity':'ethnicity', 
                        'test_preparation_course':'prep_course'}, 
                        axis=1).copy()

#menambah column average
df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# Define independent variables (factors) and target variable (performance_level)
X = df[['gender','parent_education', 'lunch', 'prep_course']]
y = df['average_score']

# Convert target variable to categorical
y = pd.cut(y, bins=[0, 40, 60, 100], labels=['Low', 'Medium', 'High'])

# Convert categorical variables to dummy variables (one-hot encoding)
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build KNeighborsClassifier model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Predict performance levels for the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Judul halaman
st.title('Prediksi Performa Siswa')

st.write('Model : KNN')
st.write('Nilai Akurasi Prediksi : ', accuracy)
st.write('')

# Pilihan gender
gender_options = ['male', 'female']
gender = st.selectbox('Gender', gender_options)

# Pilihan pendidikan orang tua
parent_education_options = ['bachelor\'s degree', 'some college', 'master\'s degree', 'associate\'s degree', 'high school', 'some high school']
parent_education = st.selectbox('Pendidikan Orang Tua (Ayah)', parent_education_options)

# Pilihan makan siang
lunch_options = ['standard', 'free/reduced']
lunch = st.selectbox('Makan Siang', lunch_options)

# Pilihan persiapan ujian
prep_course_options = ['completed', 'none']
prep_course = st.selectbox('Persiapan ujian', prep_course_options)

st.write('')

# Tombol untuk prediksi
if st.button('Prediksi Performa'):
    new_data = pd.DataFrame({
        'gender': [gender],
        'parent_education': [parent_education],
        'lunch': [lunch],
        'prep_course': [prep_course]
    })
    new_data = pd.get_dummies(new_data)

    # Menambahkan kolom-kolom yang hilang pada data baru
    missing_columns = set(X_train.columns) - set(new_data.columns)
    for col in missing_columns:
        new_data[col] = 0

    # Mengurutkan kolom sesuai dengan urutan pada saat pelatihan model
    new_data = new_data[X_train.columns]

    # Melakukan prediksi performa
    predictions = model.predict(new_data)

    # Menampilkan hasil prediksi
    st.write('')
    
    if predictions == 'High':
        st.success(f'Performa Nilai Anda : {predictions}')
    elif predictions == 'Medium':
        st.warning(f'Performa Nilai Anda : {predictions}')
    else :
        st.error(f'Performa Nilai Anda : {predictions}')
    
    if predictions == 'High':
        st.success('Rata - Rata Nilai Ujian "Matematika", "Membaca", "Menulis"  Anda Kisaran : 61 - 100')
    elif predictions == 'Medium':
        st.warning('Rata - Rata Nilai Ujian "Matematika", "Membaca", "Menulis"  Anda Kisaran : 41 - 60')
    else :
        st.error('Rata - Rata Nilai Ujian "Matematika", "Membaca", "Menulis"  Anda Kisaran : 0 - 40')

    
    st.write('Terus Tingkatkan Belajar Anda 💪!')
