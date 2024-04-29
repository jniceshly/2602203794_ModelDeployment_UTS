import pickle
import streamlit as st
import numpy as np

with open('model_pickle/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)


def make_predict(feature) :
    input_array = np.array(feature).reshape(1, -1)

    prediction = model.predict(input_array)

    return prediction

st.title("Churn Prediction")

st.header("Test Cases")

# 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
#        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'

st.subheader("First Test Case")
st.caption("Credit score: 850")
st.caption("Geography: France")
st.caption("Gender: Male")
st.caption("Age: 53")
st.caption("Tenure: 7")
st.caption("Balance: 150000")
st.caption("Number Of Products: 3")
st.caption("Has Credit Card: Yes")
st.caption("Is Active Member: Yes")
st.caption("Estimated Salary: 100000")


test_credit_score = 850
test_geograhpy = 0
test_gender = 1
test_hascc = 1
test_age = 53
test_tenure = 7
test_balance = 150000
test_product = 3
test_is_active = 1
test_estimated_salary = 100000

test_feature = [test_credit_score, test_geograhpy, test_gender, test_age, test_tenure, test_balance, test_product, test_hascc, test_is_active, test_estimated_salary]

test1_prediction = make_predict(test_feature)

test1_result = "Will Churn" if test1_prediction[0] == 1 else "Will Not Churn"

st.info(f"First Test Case Result {test1_result}")

st.subheader("Second Test Case")
st.caption("Credit score: 500")
st.caption("Geography: Germany")
st.caption("Gender: Female")
st.caption("Age: 15")
st.caption("Tenure: 5")
st.caption("Balance: 100000")
st.caption("Number Of Products: 1")
st.caption("Has Credit Card: No")
st.caption("Is Active Member: No")
st.caption("Estimated Salary: 100000")

test2_credit_score = 500
test2_geograhpy = 1
test2_gender = 0
test2_hascc = 0
test2_age = 15
test2_tenure = 5
test2_balance = 100000
test2_product = 1
test2_is_active = 0
test2_estimated_salary = 100000

test2_feature = [test2_credit_score, test2_geograhpy, test2_gender, test2_age, test2_tenure, test2_balance, test2_product, test2_hascc, test2_is_active, test2_estimated_salary]

test2_prediction = make_predict(test2_feature)

test2_result = "Will Churn" if test2_prediction[0] == 1 else "Will Not Churn"

st.info(f"Second Test Case Result {test2_result}")


st.header("Try it yourself")

credit_socre = st.slider("Credit Score", min_value=350, max_value=850, value=500)

geography = country = st.selectbox("Country", {"France", "Spain", "Germany"})
gender = st.selectbox("Gender", {"Male", "Female"})

age = st.slider("Age", min_value=0, max_value=100, value=15)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=5)
balance = st.slider("Balance", min_value=0, max_value=280000, value=100000)
number_of_product = st.slider("Number Of Product", min_value=1, max_value=4, value=1)

has_credit_card = st.checkbox("Has Credit Card")

is_active_member = st.checkbox("Is Active Member")


estimated_salary = st.slider("Estimated Salary", min_value=10, max_value=200000, value=100000)



if st.button("Make Prediction") :

    if geography == 'France' :
        geography = 0
    elif geography == 'Germany':
        geography = 1
    else :
        geography = 2

    if gender == 'Female':
        gender = 0
    else :
        gender = 1
    
    if has_credit_card : 
        has_credit_card = 1
    else :
        has_credit_card = 0

    if is_active_member :
        is_active_member = 1
    else :
        is_active_member = 0

    feature = [credit_socre, geography, gender, age, tenure, balance, number_of_product, has_credit_card, is_active_member, estimated_salary]

    prediction = make_predict(feature)

    information = ""

    if(prediction[0] == 0) :
        information = "The customer will not churn"
    else :
        information = "The customer will churn"

    st.success(f"Prediction made => {information}")





