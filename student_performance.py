import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

#Dashboard
st.title(':red[Student] Performance in Exams :bar_chart:')
st.markdown('Prediksi Performa Student dari Aribut yang ada')
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


tab1, tab2, tab3, tab4 = st.tabs(['| Data :clipboard: |', '| EDA :weight_lifter: |', '| Pertanyaan :coffee: |', '| Prediksi :chart_with_upwards_trend: |'])
with tab1:
    st.header("Student Performance Dataset")
    st.write("""
    Kumpulan data ini terdiri dari nilai yang diperoleh siswa dalam berbagai mata pelajaran. Data ini bisa digunakan untuk memahami pengaruh latar belakang orang tua, 
    persiapan ujian dll terhadap kinerja siswa
    """)
    st.write('')
    st.write(df)

# 8 klasifikasi
with tab4:
    # Define independent variables (factors) and target variable (performance_level)
    X = df[['gender','parent_education', 'lunch', 'prep_course']]
    y = df['average_score']

    # Convert target variable to categorical
    y = pd.cut(y, bins=[0, 40, 60, 100], labels=['Low', 'Medium', 'High'])

    # Convert categorical variables to dummy variables (one-hot encoding)
    X = pd.get_dummies(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Build KNeighborsClassifier model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict performance levels for the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Judul halaman
    st.title('Klasifikasi Performa Siswa')

    st.write('Model : Random Forest')
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

# EDA
with tab2:
    
    
    st.warning('##### Rename Columns')
    st.write("""
        {
            
            df.columns = df.columns.str.replace(" ", "_").copy()
            df = df.rename({'parental_level_of_education':'parent_education', 
                            'race/ethnicity':'ethnicity', 
                            'test_preparation_course':'prep_course'}, 
                            axis=1).copy()
        }
    """)
    st.write(df)

    st.write('')

    st.warning('##### Menambah Kolom Average')
    st.write("""
        {

            df['average_score'] = df[['math_score', 'reading_score', 'writing_score']].mean(axis=1)
        }
    """)
    st.write(df)

    #missing value
    # st.write('')
    # df = df.isnull().sum()
    # st.write("""
    #     {

    #         df = df.isna().sum()
    #         df
    #     }
    # """)
    # st.write(df)

    # st.write('')
    # df = df.info()
    # st.write("""
    #     {

    #         df = df.info()
    #     }
    # """)
    # st.write(df)
    st.write('')
    st.warning('Test Preparation')
    test_preparation = df.groupby('prep_course')['prep_course'].count()
    test_preparation

    plt.figure(figsize=(4,4))
    sns.countplot(data=df,x='prep_course',palette='deep')
    plt.title('Preparation Course')
    st.pyplot(plt)

    st.write('')
    st.warning('Gender')
    test_preparation = df.groupby('gender')['gender'].count()
    test_preparation

    plt.figure(figsize=(4,4))
    sns.countplot(data=df,x='gender',palette='deep')
    plt.title('Gender')
    st.pyplot(plt)

    st.write('')
    st.warning('Ethnicity')
    test_preparation = df.groupby('ethnicity')['ethnicity'].count()
    test_preparation

    plt.figure(figsize=(4,4))
    sns.countplot(data=df,x='ethnicity',palette='deep')
    plt.title('Race/Ethnicity')
    st.pyplot(plt)

    st.write('')
    st.warning('Lunch')
    test_preparation = df.groupby('lunch')['lunch'].count()
    test_preparation

    plt.figure(figsize=(4,4))
    sns.countplot(data=df,x='lunch',palette='deep')
    plt.title('Lunch')
    st.pyplot(plt)

    st.write('')
    st.warning('Parent Education')
    test_preparation = df.groupby('parent_education')['parent_education'].count()
    test_preparation

    education_order = CategoricalDtype(categories=[
        'some high school',
        'high school',
        'some college',
        'associate\'s degree',
        'bachelor\'s degree',
        'master\'s degree'
    ], ordered=True)

    # Mengkonversi kolom 'parent_education' menjadi tipe kategori dengan urutan yang ditentukan
    data_parent = df['parent_education'].astype(education_order)
    plt.figure(figsize=(4,4))
    sns.countplot(data=df,x=data_parent ,palette='deep')
    plt.title('Parent Education')
    plt.xticks(rotation=90)
    st.pyplot(plt)
    
    
    st.write('')
    st.warning('Korelasi Antara Semua Feature')
    plt.figure(figsize=(16,10),dpi=100)
    sns.heatmap(pd.get_dummies(df,drop_first=True).corr(),cmap='viridis',annot=True)
    plt.title('Correlation between features')
    st.pyplot(plt)
    
#     st.write('')
#     education_order = CategoricalDtype(categories=[
#         'some high school',
#         'high school',
#         'some college',
#         'associate\'s degree',
#         'bachelor\'s degree',
#         'master\'s degree'
#     ], ordered=True)
#     prep_mapping = {
#             'completed': 1,
#             'none': 0,
#         }
#     gender_mapping = {
#         'male': 0,
#         'female': 1,
#         }
#     lunch_mapping = {
#         'standard': 1,
#         'free/reduced': 0,
#         }
#     ethnicity_mapping = {
#         'group A' : 0,
#         'group B' : 1,
#         'group C' : 3,
#         'group D' : 4,
#         'group E' : 5
#     }
#     data_ethnicity = df['ethnicity'].map(ethnicity_mapping)
#     # data_ethnicity


#     # Gabungkan kolom-kolom kategorikal menjadi satu DataFrame
#     combined_df = pd.concat([data_lunch, data_prep, data_gender, data_parent, data_ethnicity, df['average_score']], axis=1)
#     combined_df

#     # Hitung korelasi menggunakan metode Pearson
#     correlation_matrix = combined_df.corr(method='pearson')

#     # Plot heatmap korelasi
#     plt.figure(figsize=(10, 6))
#     sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
#     plt.title('Categorical Correlation Heatmap')
#     st.pyplot(plt)
    

with tab3:
    
    # 1. Apakah ada korelasi antara latar belakang orang tua dengan kinerja siswa dalam ujian?
    st.warning('##### :red[1. Apakah ada korelasi antara latar belakang orang tua dengan kinerja siswa dalam ujian?]')

    # Mengubah latar belakang pendidikan orang tua menjadi nilai numerik
    education_mapping = {
    'some high school': 1,
    'high school': 2,
    'some college': 3,
    'associate\'s degree': 4,
    'bachelor\'s degree': 5,
    'master\'s degree': 6
    }

    data_parent = df['parent_education'].map(education_mapping)

    # Menghitung korelasi Pearson
    pearson_corr = data_parent.corr(df['average_score'], method='pearson')

    # Menghitung korelasi Spearman
    spearman_corr = data_parent.corr(df['average_score'], method='spearman')

    st.write("""
        {
        
            education_mapping = {
            'some high school': 1,
            'high school': 2,
            'some college': 3,
            'associate\'s degree': 4,
            'bachelor\'s degree': 5,
            'master\'s degree': 6
            }

            data_parent = df['parent_education'].map(education_mapping)
            pearson_corr = data_parent.corr(df['average_score'], method='pearson')
            spearman_corr = data_parent.corr(df['average_score'], method='spearman')                                
        }
    """)

    st.success(f":red[Korelasi Pearson] : {pearson_corr}")
    st.success(f":red[Korelasi Spearman] : {spearman_corr}" ) 
    

    # Membuat urutan kategori tingkat pendidikan
    education_order = CategoricalDtype(categories=[
        'some high school',
        'high school',
        'some college',
        'associate\'s degree',
        'bachelor\'s degree',
        'master\'s degree'
    ], ordered=True)

    # Mengkonversi kolom 'parent_education' menjadi tipe kategori dengan urutan yang ditentukan
    data_parent = df['parent_education'].astype(education_order)

    # Menghitung rata-rata skor berdasarkan tingkat pendidikan orang tua
    parent_edu_performance = df.groupby(data_parent)['average_score'].mean()
    # Plot bar dengan urutan tingkat pendidikan yang diinginkan
    plt.figure(figsize=(10, 6))
    sns.barplot(x=data_parent, y=df['average_score'], data=df, order=education_order.categories)
    plt.xlabel('Parent Education Level')
    plt.ylabel('Average Score')
    plt.title('Parent Education Level vs. Average Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    st.write('')
    # 2. Apakah persiapan ujian mempengaruhi kinerja siswa dalam ujian?     
    st.warning('##### :red[2. Apakah persiapan ujian mempengaruhi kinerja siswa dalam ujian?]') 

    # Mengubah latar belakang pendidikan orang tua menjadi nilai numerik
    education_mapping = {
        'completed': 1,
        'none': 0,
    }

    data_prep = df['prep_course'].map(education_mapping)

    # Menghitung korelasi Pearson dan Spearman
    pearson_corr = data_prep.corr(df['average_score'], method='pearson')
    spearman_corr = data_prep.corr(df['average_score'], method='spearman')  

    st.write("""
        {
        
            education_mapping = {
            'completed': 1,
            'none': 0,
            }

            data_prep = df['prep_course'].map(education_mapping)
            pearson_corr = data_prep.corr(df['average_score'], method='pearson')
            spearman_corr = data_prep.corr(df['average_score'], method='spearman')      
        }
    """)

    st.success(f':red[pearson_corr] : {pearson_corr}')      
    st.success(f':red[spearman_corr] : {spearman_corr}')   

    # Grafik
    prep_course_performance = df.groupby('prep_course')['average_score'].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['prep_course'], y=df['average_score'], data=df)
    plt.xlabel('Course Preparation')
    plt.ylabel('Average Score')
    plt.title('Course Preparation vs. Average Score')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    st.write('')
    # 3. Faktor-faktor apa saja mempengaruhi dan paling penting untuk kinerja siswa dalam ujian?
    st.warning('##### :red[3. Faktor-faktor apa saja :red[yang mempengaruhi] dan :red[paling penting] untuk kinerja siswa dalam ujian?]')
    
    # features = df.drop(columns=['writing_score', 'math_score', 'reading_score', 'average_score']).columns
    # features = df[['gender','parent_education', 'lunch', 'prep_course']]
    feature_names = X.columns.tolist()
    # features = pd.DataFrame(df.data, columns=df.feature_names)

    feat_img_fig = plt.figure(figsize=(6,4))
    ax1 = feat_img_fig.add_subplot(111)
    skplt.estimators.plot_feature_importances(model, feature_names=feature_names, ax=ax1, x_tick_rotation=90)
    st.pyplot(feat_img_fig, use_container_width=True)

    # factors = ['gender', 'ethnicity', 'parent_education', 'lunch', 'prep_course']
    # for factor in factors:
    #     factor_performance = df.groupby(factor)['average_score'].mean()
    #     plt.bar(factor_performance.index, factor_performance.values)
    #     plt.xlabel(factor.capitalize())
    #     plt.ylabel('Average Score')
    #     plt.title(factor.capitalize() + ' vs. Average Score')
    #     plt.xticks(rotation=90)
    #     st.pyplot(plt)
                                             
    st.write('')
    # 4. Bagaimana distribusi nilai siswa berdasarkan latar belakang orang tua dan persiapan ujian?
    st.warning('##### :red[4. Bagaimana distribusi nilai siswa berdasarkan latar belakang orang tua dan persiapan ujian?]')

    # Membuat urutan kategori tingkat pendidikan
    education_order = CategoricalDtype(categories=[
        'some high school',
        'high school',
        'some college',
        'associate\'s degree',
        'bachelor\'s degree',
        'master\'s degree'
    ], ordered=True)

    # Mengkonversi kolom 'parent_education' menjadi tipe kategori dengan urutan yang ditentukan
    data_parent2 = df['parent_education'].astype(education_order)
    # Membuat plot violin
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=data_parent2, y='average_score', hue='prep_course', data=df)
    plt.xlabel('Parent Education')
    plt.ylabel('Average Score')
    plt.title('Distribution of Student Scores by Parent Education and Prep Course')
    # plt.xticks(rotation=45)
    plt.legend(title='Prep Course')
    plt.tight_layout()
    st.pyplot(plt)

    st.write('')
    # 5. Apakah terdapat hubungan antara kinerja siswa dalam mata pelajaran tertentu dengan kinerja mereka dalam mata pelajaran lainnya?
    st.warning('##### :red[5. Apakah terdapat hubungan antara kinerja siswa dalam mata pelajaran tertentu dengan kinerja mereka dalam mata pelajaran lainnya?]')
    plt.figure(figsize=(8,5))
    sns.heatmap(pd.get_dummies(df[['writing_score','reading_score','math_score']],drop_first=True).corr(),cmap='viridis',annot=True)
    plt.title('Correlation between features')
    st.pyplot(plt)
    st.write('')


    # 6. Apakah ada korelasi antara gender dan nilai rata-rata siswa?

    st.warning('##### :red[6. Apakah ada korelasi antara gender dan nilai rata-rata siswa?]')

    # Mengubah latar belakang pendidikan orang tua menjadi nilai numerik
    education_mapping = {
    'male': 0,
    'female': 1,
    }

    data_gender = df['gender'].map(education_mapping)

    # Menghitung korelasi Pearson
    pearson_corr = data_gender.corr(df['average_score'], method='pearson')

    # Menghitung korelasi Spearman
    spearman_corr = data_gender.corr(df['average_score'], method='spearman')

    st.write("""
        {
        
            education_mapping = {
            'male': 0,
            'female': 1,
            }

            data_gender = df['gender'].map(education_mapping)
            pearson_corr = data_gender.corr(df['average_score'], method='pearson')
            spearman_corr = data_gender.corr(df['average_score'], method='spearman')                            
        }
    """)

    st.success(f":red[Korelasi Pearson] : {pearson_corr}")
    st.success(f":red[Korelasi Spearman] : {spearman_corr}" ) 

    # Menghitung rata-rata skor berdasarkan tingkat pendidikan orang tua
    gender_graf = df.groupby(data_gender)['average_score'].mean()
    # Plot bar dengan urutan tingkat pendidikan yang diinginkan
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['gender'], y=df['average_score'], data=df)
    plt.xlabel('Gender')
    plt.ylabel('Average Score')
    plt.title('Gender vs. Average Score')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    st.write('')

    # 7. Apakah kita bisa membuat model dan memprediksi dari data ini?
    st.warning('##### :red[7. Apakah kita bisa membuat model dan memprediksi dari data ini?]')

    st.write("""
        {
        
            # Define independent variables (factors) and target variable (performance_level)

            X = df[['gender','parent_education', 'lunch', 'prep_course']]
            y = df['average_score']

            # Convert target variable to categorical

            y = pd.cut(y, bins=[0, 40, 60, 100], labels=['Low', 'Medium', 'High'])

            # Convert categorical variables to dummy variables (one-hot encoding)

            X = pd.get_dummies(X)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

            # Build RandomForestClassifier model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Predict performance levels for the test data
            y_pred = model.predict(X_test)
        }
    """)

    accuracy = accuracy_score(y_test, y_pred)
    st.success(f'Nilai Akurasi Model : {accuracy}')
    st.write('')
    st.write('Confusion Matrix')
    conf_mat_fig = plt.figure(figsize=(5,3))
    ax1 = conf_mat_fig.add_subplot(111)
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax=ax1, normalize=True)
    st.pyplot(conf_mat_fig, use_container_width=True)
    st.write('')
    
  
    st.write('Classification Report')
    st.code(classification_report(y_test,y_pred))


    
