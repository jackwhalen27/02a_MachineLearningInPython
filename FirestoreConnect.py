import streamlit as st
import pandas as pd
from google.cloud import firestore
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/jwhal/Desktop/MANA Spring/Applied Machine Learning/aml-whalen-4b02c76dcd8f.json"

db = firestore.Client()

reddit = db.collection(u'reddit')
posts = list(reddit.stream())
DocsDict = list(map(lambda x: x.to_dict(), posts))
df = pd.DataFrame(DocsDict)

st.dataframe(df)

