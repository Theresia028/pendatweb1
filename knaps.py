import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib



st.title("PENAMBANGAN DATA")
st.write("Nama : Theresia Nazela")
st.write("Nim : 200411100188")
st.write("Kelas: Penambangan Data C")
upload_data, preporcessing, modeling, implementation = st.tabs(["DataSet", "Prepocessing", "Modeling", "Implementation"])


with upload_data:
    st.write("""# Upload Dataset""")
    st.write("Dataset yang digunakan saya adalah dataset history pasien pengidap diabetes. Kumpulan data ini terbuat dari tahun 2019. Dataset ini didapat dari https://www.kaggle.com/datasets/tigganeha4/diabetes-dataset-2019 yang memiliki kurang lebih 953 baris data ")
    uploaded_files = st.file_uploader("Upload file CSV Dataset ", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Dataset Anda = ", uploaded_file.name)
        st.dataframe(df)


with preporcessing:
    st.write("""# Preprocessing""")
    st.write("Data preprocessing merupakan proses perubahan data asli ke dalam bentuk data yang lebih mudah dipahami")
    st.write("Data preprocessing dilakukan untuk mempermudah proses analisis data.")
    st.title("""Dataset Diabetes""") 
    df = pd.read_csv('https://raw.githubusercontent.com/Theresia028/pendatweb1/main/diabetes_dataset__2019.csv')
    df.head()


    #Membuat variable x dimana variable label di hilangkan
    X = df.drop(columns=['Result'])

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

    #Save model normalisasi
    import joblib
    norm = "normalisasi.sav"
    joblib.dump(scaler, norm) 

    #ENCODER NILAI Y
    y = df['Result'].values
    le = preprocessing.LabelEncoder()
    le.fit(y)
    yBaru = le.transform(y)

    #SPLIT DATASET
    training, test = train_test_split(X,test_size=0.2, random_state=1)
    training_label, test_label = train_test_split(yBaru, test_size=0.2, random_state=1)

with modeling:
    st.title("""Metode dalam Klasifikasi Data Mining""") #menampilkan judul halaman 

    #membuat sub menu menggunakan checkbox
    nb = st.checkbox("Naive bayes") #chechbox naive bayes
    knn = st.checkbox("KNN")
    ds = st.checkbox("Decission Tree")

    if nb:
        anm = pd.read_csv('https://raw.githubusercontent.com/Theresia028/pendatweb1/main/diabetes_dataset__2019.csv')
        st.write(anm)

        from sklearn.metrics import make_scorer, accuracy_score,precision_score
        from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
        from sklearn.metrics import confusion_matrix

        from sklearn.model_selection import KFold,train_test_split,cross_val_score
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import train_test_split

        X=anm.iloc[:,0:4].values
        y=anm.iloc[:,4].values

        st.write("Jumlah Shape X dan Y adalah", X.shape)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        st.write("Array ", y)

        #Train and Test split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

        gaussian = GaussianNB()
        gaussian.fit(X_train, y_train)
        Y_pred = gaussian.predict(X_test) 
        accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
        acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

        cm = confusion_matrix(y_test, Y_pred)
        accuracy = accuracy_score(y_test,Y_pred)
        precision =precision_score(y_test, Y_pred,average='micro')
        recall =  recall_score(y_test, Y_pred,average='micro')
        f1 = f1_score(y_test,Y_pred,average='micro')
        st.write('Confusion matrix for Diabetes',cm)
        st.write('accuracy_Diabetes : %.3f' %accuracy)
        st.write('precision_Diabetes : %.3f' %precision)
        st.write('recall_Diabetes : %.3f' %recall)
        st.write('f1-score_Diabetes : %.3f' %f1)

    if knn:
        ami = pd.read_csv('https://raw.githubusercontent.com/Theresia028/pendatweb1/main/diabetes_dataset__2019.csv')
        st.write(ami)

        ('%matplotlib inline')
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import ListedColormap
        from sklearn import neighbors, datasets
        from sklearn.inspection import *
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        n_neighbors = 3

        # we only take the first two features. We could avoid this ugly
        # slicing by using a two-dim dataset
        X = ami.data[:, :2]
        y = ami.target

        #split dataset into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        # Create color maps
        cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
        cmap_bold = ["darkorange", "c", "darkblue"]

        for weights in ["uniform", "distance"]:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X_train,y_train)

            ax = plt.subplots()
            # DecisionBoundaryDisplay.from_estimator(
            #     clf,
            #     X,
            #     cmap=cmap_light,
            #     ax=ax,
            #     response_method="predict",
            #     plot_method="pcolormesh",
            #     xlabel=iris.feature_names[0],
            #     ylabel=iris.feature_names[1],
            #     shading="auto",
            # )

            # Plot also the training points
            sns.scatterplot(
                x=X[:, 0],
                y=X[:, 1],
                hue=ami.target_names[y],
                palette=cmap_bold,
                alpha=1.0,
                edgecolor="black",
            )
            plt.title(
                "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
            )

        plt.show()

    if ds:
        import pandas as pd
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn import tree
        from matplotlib import pyplot as plt
        from sklearn.datasets import load_iris
        from sklearn.tree import DecisionTreeClassifier
        ane = pd.read_csv('https://raw.githubusercontent.com/Theresia028/pendatweb1/main/diabetes_dataset__2019.csv')
        ane

        #Input User
        st.sidebar.header('Parameter Inputan')
        def input_user():
            Gender = st.sidebar.slider('Jenis Kelamin', 0, 1),
            Hemoglobin = st.sidebar.slider('Hemoglobin', 2.00, 18.00)
            MCH = st.sidebar.slider('MCH (Mean Cell Hemoglobin)', 14.00, 34.80)
            MCHC = st.sidebar.slider('MCHC (Mean corpuscular hemoglobin concentration)', 28.00, 34.10)
            MCV = st.sidebar.slider('MCV (Mean Cell Volume)', 70.00, 104.70)
            data = {
                'Gender' : Gender,
                'Hemoglobin': Hemoglobin,
                'MCH': MCH,
                'MCHC': MCHC,
                'MCV':MCV
            }
            fitur = pd.DataFrame(data, index = [0])

            return fitur
        input = input_user()
        st.subheader('Parameter Inputan')
        st.write(input)

        #from sklearn import preprocessing
        #normalisasi= preprocessing.normalize(input)
        #normalisasi
        #from sklearn.preprocessing import StandardScaler
        #scaler_inp = StandardScaler()
        #scaled_inp = scaler_inp.fit_transform(input)
        #features_names_inp = input.columns.copy()
        #scaled_features = pd.DataFrame(scaled, columns=features_names_inp)
        #test_d = scaled.fit_transform(input)
        #pd.DataFrame(test_d)



    # #with st.sidebar:
    #     option = st.sidebar.selectbox(
    #         'Main Submenu',
    #         ('Naive Bayes','KNN','Decission Tree','K-Means')
    #     )
    #     st.write(f"## Metode {option}")

    #     if option == 'Naive Bayes' or option == '':
    #         anm = pd.read_csv('https://raw.githubusercontent.com/dewialqurani/datamining/main/anemia.csv')
    #         anm

    #         from sklearn.metrics import make_scorer, accuracy_score,precision_score
    #         from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
    #         from sklearn.metrics import confusion_matrix

    #         from sklearn.model_selection import KFold,train_test_split,cross_val_score
    #         from sklearn.naive_bayes import GaussianNB
    #         from sklearn.model_selection import train_test_split

    #         X=anm.iloc[:,0:4].values
    #         y=anm.iloc[:,4].values

    #         st.write("Jumlah Shape X dan Y adalah", X.shape)

    #         from sklearn.preprocessing import LabelEncoder
    #         le = LabelEncoder()
    #         y = le.fit_transform(y)
    #         st.write("Array ", y)

    #         #Train and Test split
    #         X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

    #         gaussian = GaussianNB()
    #         gaussian.fit(X_train, y_train)
    #         Y_pred = gaussian.predict(X_test) 
    #         accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
    #         acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

    #         cm = confusion_matrix(y_test, Y_pred)
    #         accuracy = accuracy_score(y_test,Y_pred)
    #         precision =precision_score(y_test, Y_pred,average='micro')
    #         recall =  recall_score(y_test, Y_pred,average='micro')
    #         f1 = f1_score(y_test,Y_pred,average='micro')
    #         st.write('Confusion matrix for Anemia',cm)
    #         st.write('accuracy_Anemia : %.3f' %accuracy)
    #         st.write('precision_Anemia : %.3f' %precision)
    #         st.write('recall_Anemia : %.3f' %recall)
    #         st.write('f1-score_Anemia : %.3f' %f1)

            

    #     elif option == 'KNN':
    #         ami = pd.read_csv('https://raw.githubusercontent.com/dewialqurani/datamining/main/anemia.csv')
    #         ami

    #         ('%matplotlib inline')
    #         import pandas as pd
    #         import matplotlib.pyplot as plt
    #         import seaborn as sns
    #         from matplotlib.colors import ListedColormap
    #         from sklearn import neighbors, datasets
    #         from sklearn.inspection import *
    #         from sklearn.model_selection import train_test_split
    #         from sklearn.neighbors import KNeighborsClassifier

    #         n_neighbors = 3

    #         # we only take the first two features. We could avoid this ugly
    #         # slicing by using a two-dim dataset
    #         X = ami.data[:, :2]
    #         y = ami.target

    #         #split dataset into train and test data
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    #         # Create color maps
    #         cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
    #         cmap_bold = ["darkorange", "c", "darkblue"]

    #         for weights in ["uniform", "distance"]:
    #             # we create an instance of Neighbours Classifier and fit the data.
    #             clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    #             clf.fit(X_train,y_train)

    #             ax = plt.subplots()
    #             # DecisionBoundaryDisplay.from_estimator(
    #             #     clf,
    #             #     X,
    #             #     cmap=cmap_light,
    #             #     ax=ax,
    #             #     response_method="predict",
    #             #     plot_method="pcolormesh",
    #             #     xlabel=iris.feature_names[0],
    #             #     ylabel=iris.feature_names[1],
    #             #     shading="auto",
    #             # )

    #             # Plot also the training points
    #             sns.scatterplot(
    #                 x=X[:, 0],
    #                 y=X[:, 1],
    #                 hue=ami.target_names[y],
    #                 palette=cmap_bold,
    #                 alpha=1.0,
    #                 edgecolor="black",
    #             )
    #             plt.title(
    #                 "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    #             )

    #         plt.show()

    #     elif option == 'Decission Tree':
    #         import pandas as pd
    #         import numpy as np
    #         from sklearn.metrics import accuracy_score
    #         from sklearn import tree
    #         from matplotlib import pyplot as plt
    #         from sklearn.datasets import load_iris
    #         from sklearn.tree import DecisionTreeClassifier
    #         ane = pd.read_csv('https://raw.githubusercontent.com/dewialqurani/datamining/main/anemia.csv')
    #         ane

    #         #Input User
    #         st.sidebar.header('Parameter Inputan')
    #         def input_user():
    #             Gender = st.sidebar.slider('Jenis Kelamin', 0, 1),
    #             Hemoglobin = st.sidebar.slider('Hemoglobin', 2.00, 18.00)
    #             MCH = st.sidebar.slider('MCH (Mean Cell Hemoglobin)', 14.00, 34.80)
    #             MCHC = st.sidebar.slider('MCHC (Mean corpuscular hemoglobin concentration)', 28.00, 34.10)
    #             MCV = st.sidebar.slider('MCV (Mean Cell Volume)', 70.00, 104.70)
    #             data = {
    #                 'Gender' : Gender,
    #                 'Hemoglobin': Hemoglobin,
    #                 'MCH': MCH,
    #                 'MCHC': MCHC,
    #                 'MCV':MCV
    #             }
    #             fitur = pd.DataFrame(data, index = [0])

    #             return fitur
    #         input = input_user()
    #         st.subheader('Parameter Inputan')
    #         st.write(input)

    #         #from sklearn import preprocessing
    #         #normalisasi= preprocessing.normalize(input)
    #         #normalisasi
    #         #from sklearn.preprocessing import StandardScaler
    #         #scaler_inp = StandardScaler()
    #         #scaled_inp = scaler_inp.fit_transform(input)
    #         #features_names_inp = input.columns.copy()
    #         #scaled_features = pd.DataFrame(scaled, columns=features_names_inp)
    #         #test_d = scaled.fit_transform(input)
    #         #pd.DataFrame(test_d)


    #     elif option == 'K-Means':
    #         st.write("""## Halaman Metode K-Means""") #menampilkan judul halaman dataframe

    #         #membuat dataframe dengan pandas yang terdiri dari 2 kolom dan 4 baris data
    #         df = pd.DataFrame({
    #             'Column 1':[1,2,3,4],
    #             'Column 2':[10,12,14,16]
    #         })
    #         df #menampilkan dataframe

if selected == 'Prediksi':
    st.write("""## Hasil Prediksi""") #menampilkan judul halaman dataframe
    #Prediksi
