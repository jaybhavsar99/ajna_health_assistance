import os
import logging
from dotenv import load_dotenv
import pandas as pd
import xgboost as xgb
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Set environment variables for Langchain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Initialize Streamlit UI
st.title('Langchain Demo With LLAMA2 API')

# Input fields for user parameters
age = st.number_input("Enter your age", min_value=1, max_value=120, step=1)
gender = st.selectbox("Select your gender", ["Male", "Female", "Other"])
weight = st.number_input("Enter your weight (kg)", min_value=1, step=1)
temperature = st.number_input("Enter the surrounding temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)
rotation_rate = st.number_input("Rotation angle(degree normalized)", min_value=-3.00, max_value=3.00, step=0.1)
angley = st.number_input("(gyro angle degree normalized)", min_value=-3.00, max_value=3.00, step=0.1)
pitch = st.number_input("(pitch angle degree normalized)", min_value=-3.00, max_value=3.00, step=0.1)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a health assistant. You need to provide health feedback based on the user's activity, including information like age, gender, weight, and surrounding temperature. Ensure to offer feedback on whether the activity is safe or not and provide constructive warnings."),
        ("user", "Age: {age}, Gender: {gender}, Weight: {weight}, Temperature: {temperature}, Activity: {question}")
    ]
)

# Action dictionary to map activity names to values
action_dict = {'going down hill': 0, 'running': 1, 'siting': 2, 'standing': 3, 'climbing': 4}

# Load the model
@st.cache_resource
def load_model():
    logger.info("Loading XGBoost model...")
    model = xgb.XGBClassifier()  # Or use XGBRegressor depending on your task
    model.load_model('xgboost_model_v2.1.json')
    logger.info("Model loaded successfully.")
    return model

# Initialize the model
loaded_model = load_model()

# Define the operation triggered by button click
def process_activity():
    # Log input values
    logger.info(f"Processing with inputs: Age={age}, Gender={gender}, Weight={weight}, Temperature={temperature}, RotationRate={rotation_rate}, Angley={angley}, Pitch={pitch}")
    
    # Prepare data for prediction
    data_dict = {
        "rotationRate.y": [rotation_rate],
        "gravity.y": [angley],
        "attitude.pitch": [pitch],
    }
    df_test = pd.DataFrame(data_dict)
    
    # Make prediction
    value = loaded_model.predict(df_test)
    logger.info(f"Model prediction result: {value}")
    
    # Map prediction value to activity name
    action_name = [key for key, val in action_dict.items() if val == value]
    if not action_name:
        st.write("No matching activity found.")
        return
    
    # Log activity name
    logger.info(f"Predicted activity: {action_name[0]}")
    
    # Set up the Llama2 model
    llm = Ollama(model="llama2")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    # Provide input parameters to the chain
    response = chain.invoke({
        "age": age,
        "gender": gender,
        "weight": weight,
        "temperature": temperature,
        "question": action_name[0]
    })
    
    # Display the response
    st.write(response)
    logger.info(f"Response: {response}")

# Add a button to start the operation
if st.button('Get Health Feedback'):
    process_activity()
