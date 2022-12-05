import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# import warnings
# warnings.filterwarnings("ignore")


st.title("PENAMBANGAN DATA C")

data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["DESKRIPSI DATASET", "DATASET", "PREPROCESSING", "MODELING", "IMPLEMENTATION"])

with data_set_description:
    st.write("##### Nama  : Theresia Nazela ")
    st.write("##### Nim   : 200411100028 ")
    st.write("##### Kelas : Penambangan Data C ")
    st.write("""# Data Set Description """)
    
    st.write("Dataset yang digunakan saya adalah dataset tentang rekomendasi API Spotify. Sehingga nanti akan mendapat data informasi seperti dancebility tentang lagu favourit, dan lainnya. ")
    st.write("Data set yang saya gunakan ada 14 kolom dan terdiri dari 193 baris data. Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/bricevergnou/spotify-recommendation . ")
    st.write("Fitur - Fitur yang ada di dalam dataset ini adalah : ")
    st.write("1. Danceability : menjelaskan seberapa cocok sebuah lagu untuk menari berdasarkan kombinasi elemen musik termasuk tempo, ")
    st.write("2. Energy : adalah ukuran dari 0,0 hingga 1,0 dan mewakili ukuran perseptual intensitas dan aktivitas. Biasanya, trek energik ")
    st.write("3. Key : berada. Integer memetakan ke nada menggunakan notasi Kelas Nada standar . Misalnya. 0 = C, 1 = C♯/D♭, 2 = D, dan seterusnya")
    st.write("4. Loudness : Nilai kenyaringan dirata-ratakan di seluruh trek dan berguna ")
    st.write("5. Mode : Mode menunjukkan modalitas (mayor atau minor) dari sebuah trek, jenis tangga nada dari mana konten melodinya berasal")
    st.write("6. Speechiness : mendeteksi keberadaan kata-kata yang diucapkan di trek. ")
    st.write("7. Acousticness : Ukuran kepercayaan dari 0,0 hingga 1,0 apakah lagu tersebut akustik. 1.0 mewakili keyakinan tinggi trek tersebut ")
    st.write("8. Instrumentalness : Memprediksi apakah trek tidak berisi vokal.")
    st.write("9. Liveness : Mendeteksi kehadiran penonton dalam rekaman")
    st.write("10. Valence : A measure from 0.0 to 1.0 describes the musical positiveness conveyed by a track.")
    st.write("11. Tempo : tingkat tempo yang dihasilkan dari lagu ")
    st.write("12. Duration_ms :  durasi dari lagu ")
    st.write("13. Time_Signiture : Konvensi notasi yang digunakan dalam notasi musik ")
    st.write("14. Liked : penggemar dalam musik tersebut")
    
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/Theresia028/pendatweb1")

with upload_data:
    # uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #     df = pd.read_csv(uploaded_file)
    #     st.write("Nama File Anda = ", uploaded_file.name)
    #     st.dataframe(df)
    st.write ("DATASET ASLI")
    df = pd.read_csv('https://raw.githubusercontent.com/Theresia028/pendatweb1/main/data.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    df = df.drop(columns=['danceability','energy','loudness','speechiness','liveness','valence'])
    #Mendefinisikan Varible X dan Y
    X = df[['key','mode','acousticness','instrumentalness','tempo','duration_ms','time_signature']]
    y = df['liked'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.liked).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],

        
    })

    st.write(labels)

    # st.subheader("""Normalisasi Data""")
    # st.write("""Rumus Normalisasi Data :""")
    # st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    # st.markdown("""
    # Dimana :
    # - X = data yang akan dinormalisasi atau data asli
    # - min = nilai minimum semua data asli
    # - max = nilai maksimum semua data asli
    # """)
    # df.weather.value_counts()
    # df = df.drop(columns=["date"])
    # #Mendefinisikan Varible X dan Y
    # X = df.drop(columns=['weather'])
    # y = df['weather'].values
    # df_min = X.min()
    # df_max = X.max()

    # #NORMALISASI NILAI X
    # scaler = MinMaxScaler()
    # #scaler.fit(features)
    # #scaler.transform(features)
    # scaled = scaler.fit_transform(X)
    # features_names = X.columns.copy()
    # #features_names.remove('label')
    # scaled_features = pd.DataFrame(scaled, columns=features_names)

    # #Save model normalisasi
    # from sklearn.utils.validation import joblib
    # norm = "normalisasi.save"
    # joblib.dump(scaled_features, norm) 


    # st.subheader('Hasil Normalisasi Data')
    # st.write(scaled_features)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        key = st.number_input('Masukkan kunci lagunya (key) : ')
        mode = st.number_input('Masukkan tingkat mode lagu (mode) : ')
        acousticness = st.number_input('Masukkan tingkat akustik lagu (acousticness) : ')
        instrumentalness = st.number_input('Masukkan angka instrumen (instrumentalness) : ')
        tempo = st.number_input('Masukkan jumlah tempo lagu (tempo) : ')
        duration_ms = st.number_input('Masukkan jumlah durasi (duration_ms) : ')
        time_signature = st.number_input('Masukkan jumlah waktu notasi lagu  (time_signature) : ')
        
       
        
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                key,
                mode,
                acousticness,
                instrumentalness,
                tempo,
                duration_ms,
                time_signature
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

               
            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
