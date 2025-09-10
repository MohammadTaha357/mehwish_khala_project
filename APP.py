import streamlit  as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import  pickle 
from keras.losses import MeanAbsoluteError
## Load the trained model, scaler , onehot, label encoder

model = load_model('modelmeh.h5') # Trained Model


with open('onehot_encoder_geo.pkl','rb') as file: # Feature engineered Column Geography
    onehot_encoder_geo=pickle.load(file)

with open('lable_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

#streamlit app

st.title("Customer Churn Predictions ")


## user input 
user=st.text_input("Enter your name")
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])

gender = st.selectbox('Gender',label_encoder_gender.classes_)

age = st.slider("Age",18,100)

Balance = st.number_input("Balance",1,9999999)
credit_score = st.number_input('Credit Score',1,999)
tenure = st.slider('Tenure',0,10)
Exited = st.number_input('Exited',0,1)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


input_data= pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance' : [Balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited' : [Exited]
})


geo_encoder = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoder,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# scale the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f"Churn Probability is : {prediction_prob:.2f}")

st.write('your Estimated Salary',prediction_prob)
if user:
    st.write(f"your salary is {user}")