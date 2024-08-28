# import os
# import logging
# from flask import Flask, request, render_template
# import numpy as np
# import pandas as pd
# import tensorflow as tf

# from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# application = Flask(__name__)
# app = application

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Route for the Home page
# @app.route('/')
# def index():
#     return render_template("home.html")

# @app.route('/predictdata', methods=['GET', 'POST'])
# def predict_datapoint():
#     if request.method == 'GET':
#         return render_template('home.html')
#     else:
#         try:
#             logger.info("Collecting input data from the form")
#             data = CustomData(
#                 CreditScore=float(request.form.get('CreditScore')),
#                 Geography=request.form.get('Geography'),
#                 Gender=request.form.get('Gender'),
#                 Age=int(request.form.get('Age')),
#                 Tenure=int(request.form.get('Tenure')),
#                 Balance=float(request.form.get('Balance')),
#                 NumOfProducts=int(request.form.get('NumOfProducts')),
#                 HasCrCard=int(request.form.get('HasCrCard')),
#                 IsActiveMember=int(request.form.get('IsActiveMember')),
#                 EstimatedSalary=float(request.form.get('EstimatedSalary'))
#             )

#             pred_df = data.get_data_as_data_frame()
#             logger.info(f"Input DataFrame: {pred_df}")

#             predict_pipeline = PredictPipeline()
#             logger.info("Starting prediction pipeline")

#             results = predict_pipeline.predict(pred_df)
#             logger.info(f"Prediction results: {results}")

#             return render_template('home.html', results=results[0])

#         except Exception as e:
#             logger.error(f"Error occurred during prediction: {str(e)}")
#             return render_template('home.html', results="Error occurred during prediction. Please try again.")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", debug=True)
import os
import logging
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Route for the Home page
@app.route('/')
def index():
    return render_template("home.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect input data from the form
            logger.info("Collecting input data from the form")
            data = CustomData(
                CreditScore=float(request.form.get('CreditScore')),
                Geography=request.form.get('Geography'),
                Gender=request.form.get('Gender'),
                Age=int(request.form.get('Age')),
                Tenure=int(request.form.get('Tenure')),
                Balance=float(request.form.get('Balance')),
                NumOfProducts=int(request.form.get('NumOfProducts')),
                HasCrCard=int(request.form.get('HasCrCard')),
                IsActiveMember=int(request.form.get('IsActiveMember')),
                EstimatedSalary=float(request.form.get('EstimatedSalary'))
            )

            # Convert input data to DataFrame
            pred_df = data.get_data_as_data_frame()
            logger.info(f"Input DataFrame: {pred_df}")

            # Perform prediction
            predict_pipeline = PredictPipeline()
            logger.info("Starting prediction pipeline")

            results = predict_pipeline.predict(pred_df)
            logger.info(f"Prediction results: {results}")

            # Interpret prediction results
            predicted_class_index = np.argmax(results)  # Get index of the highest probability
            class_labels = ["No Churn", "Churn"]  # Adjust labels according to your model
            prediction_label = class_labels[predicted_class_index]

            logger.info(f"Prediction Label: {prediction_label}")
            return render_template('home.html', results=prediction_label)

        except Exception as e:
            logger.error(f"Error occurred during prediction: {str(e)}")
            return render_template('home.html', results="Error occurred during prediction. Please try again.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
