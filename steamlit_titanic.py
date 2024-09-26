import streamlit as st
import joblib
import numpy as np


model=joblib.load(r'C:\Users\mwael\OneDrive\Desktop\after_cource\projects\titanic\saved_model\decision_tree_model.pkl')


st.title('Titanic Survival Prediction')


st.header('Please provide the following information:')


pclass = st.selectbox('Passenger Class (Pclass)', (1, 2, 3))
sex = st.selectbox('Sex', ('Male', 'Female'))
sibsp = st.number_input('Number of Siblings/Spouses aboard (SibSp)', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children aboard (Parch)', min_value=0, max_value=10, value=0)
family = sibsp+parch
fare = st.slider('Fare', 10.0, 500.0, 1.0)
embarked = st.selectbox('Port of Embarkation (Embarked)', ('Southampton', 'Cherbourg', 'Queenstown'))


sex = 1 if sex == 'Male' else 0

embarked_dict = {'Southampton': 2, 'Cherbourg': 0, 'Queenstown': 1}
embarked = embarked_dict[embarked]


features = np.array([[pclass, sex, sibsp, parch, fare, embarked,family]])


if st.button('Predict'):
    prediction = model.predict(features) 

    if prediction[0] == 1:
        st.success('This person would have survived!')
    else:
        st.error('This person would not have survived.')









