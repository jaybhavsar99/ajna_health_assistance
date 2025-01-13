import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


loaded_model = xgb.XGBClassifier()  # Or use XGBRegressor depending on your task
loaded_model.load_model('xgboost_model_new.json')



data_dict = {
    "attitude.roll": [0.461886],
    "attitude.pitch": [-0.064767],
    "attitude.yaw": [1.191620],
    "gravity.x": [0.444703],
    "gravity.y": [0.064722],
    "gravity.z": [-0.893337],
    "rotationRate.x": [0.003074],
    "rotationRate.y": [-0.004155],
    "rotationRate.z": [0.002212],
    "userAcceleration.x": [0.003448],
    "userAcceleration.y": [-0.005487],
    "userAcceleration.z": [-0.002934],
}
y_pred1 = loaded_model.predict(df)

predicted_class_names = label_encoder.inverse_transform(y_pred1)
