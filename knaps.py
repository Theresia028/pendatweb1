# import numpy as np
# import pandas as pd
# # import seaborn as sns
# import sklearn
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import sklearn
import streamlit as st
import pandas as pd 
import numpy as np 
import warnings
from sklearn.metrics import make_scorer, accuracy_score,precision_score
warnings.filterwarnings('ignore', category=UserWarning, append=True)

# data
df = pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/machine-Learning/main/mushrooms.csv")
df.head()

# normalisasi
# data yang dipakai 2000 data
# pemisahan class dan fitur
df=df[:2000]
from sklearn.preprocessing import OrdinalEncoder
x = df.drop(df[['class']],axis=1)
enc = OrdinalEncoder()
a = enc.fit_transform(x)
x=pd.DataFrame(a, columns=x.columns)

# class
y = df.loc[:, "class"]
y = df['class'].values

# Split Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

st.set_page_config(page_title="Ima")
@st.cache()
def progress():
    with st.spinner("Bentar ya....."):
        time.sleep(1)
        
st.title("UAS PENDAT")

dataframe, preporcessing, modeling, implementation = st.tabs(
    ["Jamur Data", "Prepocessing", "Modeling", "Implementation"])

with dataframe:
    st.write('Data Jamur')
    dataset,data= st.tabs(['Dataset',"data"])
    with dataset:
        st.dataframe(df)

        
with preporcessing:
    st.write('Ordinal Encoder')
    st.dataframe(x)

with modeling:
    # pisahkan fitur dan label
    knn,lainnya= st.tabs(
        ["K-Nearest Neighbor","lainnya"])
    with knn:
      from sklearn.neighbors import KNeighborsClassifier
      knn = KNeighborsClassifier(n_neighbors=3)
      knn.fit(x_train,y_train)
      y_pred_knn = knn.predict(x_test) 
      accuracy_knn=round(accuracy_score(y_test,y_pred_knn)* 100, 2)
      acc_knn = round(knn.score(x_train, y_train) * 100, 2)
      label_knn = pd.DataFrame(
      data={'Label Test': y_test, 'Label Predict': y_pred_knn}).reset_index()
      st.success(f'Tingkat akurasi = {acc_knn}')
      st.dataframe(label_knn)

with implementation:
      df=df[:2000]
      from sklearn.preprocessing import OrdinalEncoder
      x = df.drop(df[['class']],axis=1)
      enc = OrdinalEncoder()
      a = enc.fit_transform(x)
      x=pd.DataFrame(a, columns=x.columns)
#       x_new = ['x','y','y','t','l','f','c','b','g','e','c','s','s','w','w','p','w','o','p','k','s','m'] # hasil=0/e
      x_new = ["x","s","w","t","p","f","c","n","k","e","e","s","s","w","w","p","w","o","p","k","v","g"] # hasil=1/p
      hinput=enc.transform(np.array([x_new]))
      hinput
      from sklearn.neighbors import KNeighborsClassifier
      knn = KNeighborsClassifier(n_neighbors=3)
      knn.fit(x_train,y_train)
      Y_pred = knn.predict(x_test) 
      accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)
      acc_knn = round(knn.score(x_train, y_train) * 100, 2)
      accuracy_knn
      acc_knn
      y_predict = knn.predict(hinput)
      st.write("Hasil prediksi adalah",y_predict[0])
      # return y_predict[0]

# # KNN
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train,y_train)

# # Akurasi
# Y_pred = knn.predict(x_test) 
# accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)
# acc_knn = round(knn.score(x_train, y_train) * 100, 2)
# accuracy_knn
# acc_knn

# x_new = ['x','y','y','t','l','f','c','b','g','e','c','s','s','w','w','p','w','o','p','k','s','m'] # hasil=0/e
# # x_new = ["x","s","w","t","p","f","c","n","k","e","e","s","s","w","w","p","w","o","p","k","v","g"] # hasil=1/p
# hinput=enc.transform(np.array([x_new]))
# hinput

# def KNN(x_new):
#       from sklearn.neighbors import KNeighborsClassifier
#       knn = KNeighborsClassifier(n_neighbors=3)
#       knn.fit(x_train,y_train)
#       Y_pred = knn.predict(x_test) 
#       accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)
#       acc_knn = round(knn.score(x_train, y_train) * 100, 2)
#       accuracy_knn
#       acc_knn
#       y_predict = knn.predict(x_new)
#       print(y_predict[0])
#       return y_predict[0]
# KNN(hinput)
