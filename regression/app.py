import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = pickle.load(open('logistic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title='Smart Profit Predictor', layout='wide')

st.title('🛒 Smart Profit Prediction System')
st.markdown('### AI-Powered E-commerce Profit Analysis Dashboard')

# Sidebar
st.sidebar.header('Input Features')

quantity = st.sidebar.slider('Quantity', 1, 10, 3)
sales = st.sidebar.number_input('Sales', 100, 10000, 2000)
profit = st.sidebar.number_input('Profit', 10, 1000, 200)
category = st.sidebar.selectbox('Category', ['Accessories','Electronics','Office'])
region = st.sidebar.selectbox('Region', ['North','South','East','West'])

# Encoding manually
category_map = {
    'Accessories':0,
    'Electronics':1,
    'Office':2
}

region_map = {
    'East':0,
    'North':1,
    'South':2,
    'West':3
}

input_data = pd.DataFrame({
    'Product Name':[0],
    'Category':[category_map[category]],
    'Region':[region_map[region]],
    'Quantity':[quantity],
    'Sales':[sales],
    'Profit':[profit],
    'Year':[2024],
    'Month':[1]
})

scaled_data = scaler.transform(input_data)

prediction = model.predict(scaled_data)
pred_prob = model.predict_proba(scaled_data)

st.subheader('Prediction Result')

if prediction[0] == 1:
    st.success('✅ High Profit Order Predicted')
else:
    st.error('❌ Low Profit Order Predicted')
st.write('Prediction Probability:', pred_prob)

# UNIQUE FEATURE 1
st.subheader('📊 Live Business Insights')

chart_data = pd.DataFrame({
    'Metrics':['Sales','Profit'],
    'Values':[sales, profit]
})

fig, ax = plt.subplots()
sns.barplot(x='Metrics', y='Values', data=chart_data, ax=ax)
st.pyplot(fig)

# UNIQUE FEATURE 2
st.subheader('💡 AI Suggestions')

if sales > 5000:
    st.info('High sales detected. Consider increasing stock.')
else:
    st.warning('Sales are moderate. Run marketing campaigns.')
if profit < 100:
    st.warning('Profit margin is low. Reduce operational cost.')
else:
    st.success('Healthy profit margin observed.')

# UNIQUE FEATURE 3
st.subheader('📈 Business Health Meter')

health_score = min((profit/sales)*100*10, 100)

st.progress(int(health_score))

st.write(f'Business Health Score: {health_score:.2f}/100')

# UNIQUE FEATURE 4
st.subheader('🔍 Feature Importance')

importance = pd.DataFrame({
    'Features':input_data.columns,
    'Importance':abs(model.coef_[0])
})

importance = importance.sort_values(by='Importance', ascending=False)
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.barplot(x='Importance', y='Features', data=importance, ax=ax2)
st.pyplot(fig2)