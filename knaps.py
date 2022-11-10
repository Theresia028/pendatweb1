import streamlit as st
import re
import pandas as pd 
import numpy as np

st.title("""
FORM INPUT DATA SCORE
""")

#Fractional Knapsack Problem
#Getting input from user
kodekontrak=int(st.number_input("Kode Kontrak: "))
Pendapat=int(st.number_input("Pendapatan Setahun : "))
durasipinjaman=int(st.number_input("Durasi Pinjaman : "))
jumlahtanggungan=int(st.number_input("Jumlah Tanggungan : "))
kpr=str(st.text_input("KPR : ",'ya'))
ovd=str(st.text_input("Rata-Rata Overdue : ",'ya'))

submit = st.button("submit")


if submit:
    st.info("Jadi,dinyakataan . ")


