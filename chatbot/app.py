from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import xgboost as xgb
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
## Prompt Template


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a health assistant. You need to provide health feedback based on the user's activity, including information like age, gender, weight, and surronding temperature. Ensure to offer feedback on whether the activity is safe or not and provide constructive warnings."),
        ("user", "Age: {age}, Gender: {gender}, Weight: {weight}, Temperature: {temperature}, Activity: {question}")
    ]
)

## Streamlit framework
st.title('Langchain Demo With LLAMA2 API')

# Input fields for user parameters
age = st.number_input("Enter your age", min_value=1, max_value=120, step=1)
gender = st.selectbox("Select your gender", ["Male", "Female", "Other"])
weight = st.number_input("Enter your weight (kg)", min_value=1, step=1)
temperature = st.number_input("Enter the surrounding temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)
rotation_rate = st.number_input("Rotation angle(degree normalized)", min_value=-3.00, max_value=3.00, step=0.1)
angley = st.number_input("(gyro angledegree normalized)", min_value=-3.00, max_value=3.00, step=0.1)
pitch = st.number_input("(pitch angledegree normalized)", min_value=-3.00, max_value=3.00, step=0.1)

data_dict = {
    "rotationRate.y": [rotation_rate],
    "gravity.y": [angley],
    "attitude.pitch": [pitch],
}
df_test = pd.DataFrame(data_dict)

loaded_model = xgb.XGBClassifier()  # Or use XGBRegressor depending on your task
loaded_model.load_model('xgboost_model_v2.1.json')

value = loaded_model.predict(df_test)
action_dict = {'going down hill': 0, 'running': 1, 'siting': 2, 'standing': 3, 'clibing': 4}
action_name = [key for key, val in action_dict.items() if val == value]
#predicted_class_names = label_encoder.inverse_transform(y_pred1)
#input_text = st.text_input("Search the activity or question you want feedback on")

# Ollama Llama2 LLM
llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if action_name:
    # Provide input parameters to the chain
    response = chain.invoke({
        "age": age,
        "gender": gender,
        "weight": weight,
        "temperature": temperature,
        "question": action_name
    })
    
    # Display the response
    st.write(response)
