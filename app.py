import joblib
from skimage.feature import hog
import streamlit as st
from PIL import Image
import time
import numpy as np

model = joblib.load("hog_mlp_classifier.joblib")
mean_var_scaler = joblib.load("val_scaler.joblib")
ActionText = {0 : 'SafeDriving', 1 : 'TextingRight', 2 : 'CellphoneTalkingRight', 3 : 'TextingLeft', 
              4 : 'CellphoneTalkingLeft', 5 : 'OperatingRadio', 6 : 'Drinking', 7 : 'ReachingBehind', 
              8 : 'SelfGrooming', 9 : 'TalkingToOthers'}

st.title("Driver Distraction detection")
uploaded_file = st.file_uploader("Upload a file")
col1, col2 = st.columns([0.7,0.3])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    with col1:
        st.image(img,width=470)  
    start = time.time()
    img = img.convert("L")
    img = img.resize((240,320), Image.NEAREST)
    x = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
    x = mean_var_scaler.transform(np.expand_dims(x,axis=0))
    y = model.predict_proba(x)
    end = time.time()
    y1, y2 = np.argsort(y[0])[::-1][:2]
    y1, y2 = ActionText[y1], ActionText[y2]
    print(y1,y2)
    with col2:
        st.write(f"first proabable action - \"{y1}\"")
        st.write(f"second proabable action - \"{y2}\"")
        st.write(f"time taken to find the Action - {(end-start)*10**3:.03f}ms")