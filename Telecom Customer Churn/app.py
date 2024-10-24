import pickle
import streamlit as st
import pandas as pd
from PIL import Image

# Load the saved model and DictVectorizer
model_file = 'Customer Churn.pkl'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Streamlit app for customer churn prediction
def main():
    image = Image.open('images (2).png')
    image2 = Image.open('images.png')
    st.image(image, use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))
    st.sidebar.info('Business app to predict Customer Churn')
    st.sidebar.image(image2)
    st.title("Predicting Customer Churn")

    # Online prediction
    if add_selectbox == 'Online':
        gender = st.selectbox('Gender:', ['male', 'female'])
        seniorcitizen = st.selectbox('Customer is a senior citizen:', [0, 1])
        partner = st.selectbox('Customer has a partner:', ['yes', 'no'])
        dependents = st.selectbox('Customer has dependents:', ['yes', 'no'])
        phoneservice = st.selectbox('Customer has phone service:', ['yes', 'no'])
        multiplelines = st.selectbox('Customer has multiple lines:', ['yes', 'no', 'no_phone_service'])
        internetservice = st.selectbox('Customer has internet service:', ['dsl', 'no', 'fiber_optic'])
        onlinesecurity = st.selectbox('Customer has online security:', ['yes', 'no', 'no_internet_service'])
        onlinebackup = st.selectbox('Customer has online backup:', ['yes', 'no', 'no_internet_service'])
        deviceprotection = st.selectbox('Customer has device protection:', ['yes', 'no', 'no_internet_service'])
        techsupport = st.selectbox('Customer has tech support:', ['yes', 'no', 'no_internet_service'])
        streamingtv = st.selectbox('Customer has streaming TV:', ['yes', 'no', 'no_internet_service'])
        streamingmovies = st.selectbox('Customer has streaming movies:', ['yes', 'no', 'no_internet_service'])
        contract = st.selectbox('Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
        paperlessbilling = st.selectbox('Customer has paperless billing:', ['yes', 'no'])
        paymentmethod = st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check', 'mailed_check'])
        tenure = st.number_input('Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
        monthlycharges = st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
        totalcharges = tenure * monthlycharges
        output = ""
        output_prob = ""

        input_dict = {
            "gender": gender,
            "seniorcitizen": seniorcitizen,
            "partner": partner,
            "dependents": dependents,
            "phoneservice": phoneservice,
            "multiplelines": multiplelines,
            "internetservice": internetservice,
            "onlinesecurity": onlinesecurity,
            "onlinebackup": onlinebackup,
            "deviceprotection": deviceprotection,
            "techsupport": techsupport,
            "streamingtv": streamingtv,
            "streamingmovies": streamingmovies,
            "contract": contract,
            "paperlessbilling": paperlessbilling,
            "paymentmethod": paymentmethod,
            "tenure": tenure,
            "monthlycharges": monthlycharges,
            "totalcharges": totalcharges
        }

        if st.button("Predict"):
            # Transform the input data
            X = dv.transform([input_dict])
            # Make prediction
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
        
        st.success(f'Churn Prediction: {output}')
        if isinstance(output_prob, (int, float)):  # Check if output_prob is numeric
            st.success('Risk Score: {}%'.format(int(output_prob * 100)))


    # Batch prediction
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            X = dv.transform(data.to_dict(orient='records'))
            y_pred = model.predict_proba(X)[:, 1]
            churn_pred = y_pred >= 0.5
            data['Churn Prediction'] = churn_pred
            data['Churn Probability'] = y_pred
            st.write(data)


if __name__ == '__main__':
    main()
