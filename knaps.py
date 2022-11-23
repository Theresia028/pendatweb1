import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn import svm
from sklearn import metrics 

st.title(' Aplikasi Web Data Mining')
st.write("""
### Klasifikasi pengaruh Covid-19 terhadap Liver
""")
st.write('## 1. Introduction')
st.sidebar.write("""
            ### Pilih Metode yang anda inginkan :"""
            )
algoritma=st.sidebar.selectbox(
    'Pilih', ('Decision Tree','Random Forest','SVM')
)

st.write('## 2. About Dataset (Heart Failure)')
data_hf = pd.read_csv("https://raw.githubusercontent.com/Theresia028/pendatweb1/main/covid-liver.csv")
st.write("Dataset Covid1- Liver : (https://raw.githubusercontent.com/Theresia028/pendatweb1/main/covid-liver.csv) ", data_hf)

st.write('Dataset Description :')
st.write('1. Cancer: (Y)')
st.write('2. Year: (Prepandemic/Pandemic)')
st.write('3. Month: (1 - 12))')
st.write('4. Bleed:(N)')
st.write('5. Mode_Presentation: Percentage of blood leaving the heart at each contraction')
st.write('6. Age: If the patient has hypertension(Boolean)')
st.write('7. Gender: Platelet count of blood (kiloplatelets/mL)')
st.write('8. Etiology: Level of serum creatinine in the blood (mg/dL)')
st.write('9. Cirrhosis: Level of serum sodium in the blood (mEq/L)')
st.write('10. Size: Sex of the patient(Boolean)')
st.write('11. HCC_TNM_Stage: If the patient smokes or not(Boolean)')
st.write('12. HCC_BCLC_Stage: Follow-up period')
st.write('13. ICC_TNM_Stage: If the patient deceased during the follow-up period')
st.write('14. Treatment_grps: If the patient deceased during the follow-up period')
st.write('15. Survival_fromMDM: If the patient deceased during the follow-up period')
st.write('16. Alive_Dead: If the patient deceased during the follow-up period')
st.write('17. Type_of_incidental_finding: If the patient deceased during the follow-up period')
st.write('18. Surveillance_programme: If the patient deceased during the follow-up period')
st.write('19. Surveillance_effectiveness: If the patient deceased during the follow-up period')
st.write('20. Mode_of_surveillance_detection: If the patient deceased during the follow-up period')
st.write('21. Time_diagnosis_1st_Tx: If the patient deceased during the follow-up period')
st.write('22. Date_incident_surveillance_scan: If the patient deceased during the follow-up period')
st.write('23. PS: If the patient deceased during the follow-up period')
st.write('24. Time_MDM_1st_treatment: If the patient deceased during the follow-up period')
st.write('25. Time_decisiontotreat_1st_treatment: If the patient deceased during the follow-up period')
st.write('26. Prev_known_cirrhosis: If the patient deceased during the follow-up period')
st.write('27. Months_from_last_surveillance: If the patient deceased during the follow-up period')
 
st.write('Jumlah baris dan kolom :', data_hf.shape)

X=data_hf.iloc[:,0:12].values 
y=data_hf.iloc[:,12].values

st.write('Jumlah kelas : ', len(np.unique(y)))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


#Train and Test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# Decision Tree
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, y_train)  
Y_pred = decision_tree.predict(X_test) 
accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for DecisionTree\n',cm)
print('accuracy_DecisionTree: %.3f' %accuracy)
print('precision_DecisionTree: %.3f' %precision)
print('recall_DecisionTree: %.3f' %recall)
print('f1-score_DecisionTree : %.3f' %f1)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)
accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)


cm = confusion_matrix(y_test, Y_prediction)
accuracy = accuracy_score(y_test,Y_prediction)
precision =precision_score(y_test, Y_prediction,average='micro')
recall =  recall_score(y_test, Y_prediction,average='micro')
f1 = f1_score(y_test,Y_prediction,average='micro')
print('Confusion matrix for Random Forest\n',cm)
print('accuracy_random_Forest : %.3f' %accuracy)
print('precision_random_Forest : %.3f' %precision)
print('recall_random_Forest : %.3f' %recall)
print('f1-score_random_Forest : %.3f' %f1)

#SVM
SVM = svm.SVC(kernel='linear') 
SVM.fit(X_train, y_train)
Y_prediction = SVM.predict(X_test)
accuracy_SVM=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_SVM = round(SVM.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for SVM\n',cm)
print('accuracy_SVM : %.3f' %accuracy)
print('precision_SVM : %.3f' %precision)
print('recall_SVM : %.3f' %recall)
print('f1-score_SVM : %.3f' %f1)

st.write('## 3. Akurasi Metode')

results = pd.DataFrame({
    'Model': ['Decision Tree','Random Forest','SVM'],
    'Accuracy_score':[accuracy_dt,
                      accuracy_rf,accuracy_SVM
                     ]})
result_df = results.sort_values(by='Accuracy_score', ascending=False)
result_df = result_df.reset_index(drop=True)
result_df.head(9)
st.write(result_df)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(['Decision Tree', 'Random Forest','SVM'],[accuracy_dt, accuracy_rf, accuracy_SVM])
plt.show()
st.pyplot(fig)

age = st.sidebar.number_input("umur =", min_value=40 ,max_value=90)
anemia = st.sidebar.number_input("anemia =", min_value=0, max_value=1)
creatinine_phosphokinase = st.sidebar.number_input("creatinine_phosphokinase =", min_value=0 , max_value=10000)
diabetes = st.sidebar.number_input("diabetes =", min_value=0, max_value=1)
ejection_fraction = st.sidebar.number_input("ejection_fraction =", min_value=0, max_value=100)
high_blood_pressure = st.sidebar.number_input("high_blood_pressure =", min_value=0 ,max_value=1)
platelets = st.sidebar.number_input("platelets =", min_value=100000, max_value=1000000)
serum_creatinine = st.sidebar.number_input("serum_creatinine =", min_value=0, max_value=10)
serum_sodium = st.sidebar.number_input("serum_sodium =", min_value=100, max_value=150)
sex = st.sidebar.number_input("sex =", min_value=0, max_value=1)
smoking = st.sidebar.number_input("smoking =", min_value=0, max_value=1)
time = st.sidebar.number_input("time =", min_value=1, max_value=500)
submit = st.sidebar.button("Submit")


