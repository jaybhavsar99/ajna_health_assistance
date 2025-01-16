Health Assistance System Implementation

Overview:
This system provides constructive feedback to users based on the activities they perform. The feedback is generated using data from the user's mobile phone, machine learning classification, and AI models, leading to personalized health suggestions.

Implementation Workflow:

1. Data Collection:
   - The system collects data from a mobile phone's sensors (accelerometer, gyroscope) to track the user's activity.
   - Data Source: Open-source motion-sense dataset (available at Kaggle: https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset).

2. Activity Classification Using ML Model:
   - A multi-class classifier (using XGBoost) is trained on the sensor data (gyro.y, pitch.y, and rotation rate) to classify the user's activity into different categories (e.g., walking, running, sitting, etc.).

3. Trigger Parameters for Feedback Generation:
   - The classified activity, along with user details like age, gender, weight, location, and temperature, are passed as trigger parameters to the AI feedback generation system.

4. AI Feedback Generation (GenAI):
   - Llama 2 Model: The system uses the Llama 2 model implemented locally using the Ollama wrapper to generate feedback.
   - LangChain: LangChain is used to facilitate interaction with the GenAI model and manage the execution flow.

5. Human-like Feedback Response:
   - The generated feedback is human-like, providing health assistance to the user based on their current activity.

Technical Implementation:

Data Source:
- Motion-Sense Dataset: This open-source dataset is used to classify human activities based on motion sensor data (gyroscope and accelerometer).
- Motion-Sense Dataset on Kaggle: https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset

Activity Classification Model:
- Model Type: Multi-class classifier built using the XGBoost algorithm.
- Input Parameters:
- Gyro.y: Angle of inclination on the y-axis (from the gyroscope).
- Pitch.y: Altitude data (from the accelerometer).
- Rotation rate: Motion rate on the y-axis.

AI Model and Feedback Generation (GenAI):
- LangChain: Utilized to manage the model's input and output.
- Streaming Model Parameters: LangChain streams model parameters to monitor the performance and execution of the model.
- Ollama Wrapper: Implements the Llama 2 model locally for generating feedback.

UI and Backend:
- Streamlit: Used for the frontend UI where users can interact with the system.
- Backend Operations: Handled by Streamlit’s backend, which interfaces with the AI model, classifier, and LangChain.

File Structure:

project_root/
├── requirements.txt          # Contains all required libraries and packages
├── chatbot/
│   ├── app_main.py           # Main execution file for running the Streamlit app
├── action_model_code/
│   ├──             # Code for training the XGBoost classifier for activity classification
└── README.md                 # Documentation and setup instructions

Execution Flow:

1. Setup Virtual Environment:
   - Create and activate a virtual environment for the project:
     python -m venv Ajna_venv  # Create virtual environment
     Ajna_venv\Scripts\activate  # For Windows

2. Install Required Packages:
   - Install necessary libraries from the requirements.txt file:
     pip install -r requirements.txt

3. Update Access Keys and Tokens:
   - Update your LangChain API key and other necessary tokens (such as Ollama API key) in the app_main.py file.
     # In app_main.py, update the keys like:
     langchain_api_key = "your-langchain-api-key"

4. Setup Ollama Locally:
   - Install Ollama locally by following the official Ollama setup instructions.
   - Ensure that Llama 2 is set as the active model in Ollama:
     ollama run llama2

5. Run Streamlit Application:
   - Execute the app_main.py file to start the Streamlit application:
     streamlit run chatbot/app_main.py

6. Access the UI:
   - The Streamlit app will be hosted locally, and you can access it through the provided local IP (e.g., http://localhost:8501).

7. Interact with the Health Assistance System:
   - Users can provide their activity data, and the system will classify the activity, generate human-like feedback, and display it through the UI.

Summary of Steps:
1. Setup Virtual Environment: Install dependencies from requirements.txt.
2. Update API Keys: Set LangChain and Ollama API keys in app_main.py.
3. Install Ollama and Llama 2: Setup Ollama and Llama 2 model locally.
4. Run Streamlit App: Launch the health assistance app using Streamlit.
5. Interact with Feedback System: Users provide activity data and receive feedback.
