import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import recall_score, precision_score, f1_score, fbeta_score, accuracy_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model  # Import the evaluation function

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_model(self, hp):
        model = Sequential()
    
        lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
        use_bidirectional = hp.Boolean('use_bidirectional')

        # Set the input_shape for the LSTM layer
        input_shape = (self.X_train.shape[1], 1)  # Assuming the data is reshaped to (timesteps, features)

        if use_bidirectional:
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(0.5))
            model.add(Bidirectional(LSTM(lstm_units // 2)))
        else:
            model.add(LSTM(lstm_units, return_sequences=False, input_shape=input_shape))

        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(self.y_train_categorical.shape[1], activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
        return model
    


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            self.X_train, self.y_train, self.X_test, self.y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Preprocess target variable for categorical classification
            self.y_train_categorical = to_categorical(self.y_train)
            self.y_test_categorical = to_categorical(self.y_test)

            # Initialize the tuner
            tuner = kt.Hyperband(
                self.build_model,
                objective='val_accuracy',
                max_epochs=10,
                factor=3,
                directory='my_dir',
                project_name='intro_to_kt'
            )

            # Early stopping callback
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

            logging.info("Starting hyperparameter search")
            tuner.search(self.X_train, self.y_train_categorical, epochs=10, validation_split=0.2, callbacks=[stop_early])

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            logging.info(f"Best LSTM units: {best_hps.get('lstm_units')}")
            logging.info(f"Best Bidirectional: {best_hps.get('use_bidirectional')}")
            logging.info(f"Best Learning Rate: {best_hps.get('learning_rate')}")

            best_model = tuner.hypermodel.build(best_hps)

            # Train the best model
            logging.info("Training the best model")
            history = best_model.fit(self.X_train, self.y_train_categorical, 
                                     epochs=10, 
                                     batch_size=32, 
                                     validation_split=0.2, 
                                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

            # Save the best model
            logging.info("Saving the best model")
            best_model.save(self.model_trainer_config.trained_model_file_path)

            # Evaluate the model on the test set
            logging.info("Evaluating the best model")
            y_pred = best_model.predict(self.X_test)
            y_pred_labels = np.argmax(y_pred, axis=1)  # Convert predictions to label indices

            evaluation_metrics = evaluate_model(self.y_test, y_pred_labels)
            logging.info(f"Evaluation metrics: {evaluation_metrics}")

            return best_model, history, evaluation_metrics

        except Exception as e:
            raise CustomException(e, sys)
