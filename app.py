import streamlit as st
import pandas as pd
import joblib

# Function to load models and encoders
def load_resources():
    model_path = "path/to/randomforest.pkl"  # Change to your actual path
    encoder_path = "label_encoder.pkl"  # Change to your actual path
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder

model, label_encoder = load_resources()

# Setting up the Streamlit app
st.title('Parking Violation Prediction')
st.write("Enter the details to predict the parking violation code:")

# Creating input fields for the user
fine = st.number_input('Enter the fine amount', min_value=0.0, format="%.2f")
year = st.number_input('Enter the year', min_value=2000, max_value=2050, step=1)
month = st.selectbox('Select the month', list(range(1, 13)))
day = st.selectbox('Select the day', list(range(1, 32)))

# Button to make predictions
if st.button('Predict Violation'):
    input_data = pd.DataFrame([[fine, year, month, day]], columns=['Fine', 'Year', 'Month', 'Day'])
    try:
        prediction = model.predict(input_data)
        decoded_violation = label_encoder.inverse_transform(prediction)[0]
        st.write(f'Predicted Violation: {decoded_violation}')
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Use this to run the Streamlit app (uncomment when running outside of a script environment)
# if __name__ == '__main__':
#     st.run()
