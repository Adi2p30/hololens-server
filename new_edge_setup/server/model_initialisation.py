import json
import torch
import os
import logging
from ultralytics import YOLO
import cv2

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Model():
    def __init__(self, model_type: str, path: str, device: str = 'cuda', timeout: float = 0.1, priority: int = 0):
        self.model_type = model_type
        self.path = path
        self.device = device
        if priority < 0:
            logger.error(f"Priority must be a non-negative integer, got {priority}.")
            raise ValueError("Priority must be a non-negative integer.")
        self.priority = priority
        self.timeout = timeout
        self.model = None

        dir_path = os.path.dirname(self.path)
        if dir_path and not os.path.exists(dir_path):
            logger.warning(f"Directory for model path '{self.path}' does not exist: '{dir_path}'. Create it if saving is intended.")
        elif not os.path.exists(self.path):
            logger.warning(f"Model path '{self.path}' does not exist. Loading will likely fail.")

    def load_model(self):
        logger.info(f"Attempting to load model type '{self.model_type}' from path '{self.path}' onto device '{self.device}'...")
        try:
            if self.model_type == 'ultralytics_yolo':
                if not os.path.exists(self.path):
                     logger.error(f"YOLO model file not found at {self.path}")
                     raise FileNotFoundError(f"YOLO model file not found at {self.path}")

                self.model = YOLO(self.path)

                logger.info(f"Model '{self.model_type}' loaded successfully from {self.path}.")

            elif self.model_type == 'VLM':
                 logger.warning(f"Loading logic for model type '{self.model_type}' is not implemented yet.")
                 return None

            else:
                logger.error(f"Loading implementation for model type '{self.model_type}' is not available.")
                return None

        except FileNotFoundError as fnf_error:
             logger.error(f"Failed to load model: {fnf_error}")
             self.model = None
             return None
        except Exception as e:
            logger.error(f"Error loading model {self.model_type} from {self.path}: {e}", exc_info=True)
            self.model = None
            return None

        return self.model

    def predict(self, data):
        if self.model is None:
            logger.error(f"Model '{self.model_type}' is not loaded. Cannot predict.")
            return None

        try:
            if self.model_type == 'ultralytics_yolo':
                logger.debug(f"Running YOLO prediction with data type: {type(data)}")
                results = self.model.predict(data, device=self.device, verbose=False)
                return results

            elif self.model_type == 'VLM':
                 logger.warning(f"Prediction logic for model type '{self.model_type}' is not implemented yet.")
                 return f"Processed VLM data: {str(data)[:50]}..."

            else:
                logger.error(f"Prediction logic for model type '{self.model_type}' is not available.")
                return None

        except Exception as e:
            logger.error(f"Error during prediction with model {self.model_type}: {e}", exc_info=True)
            return None

    def save_model(self, save_path):
        if self.model is None:
            logger.warning(f"Model '{self.model_type}' is not loaded. Cannot save.")
            return

        if not os.path.exists(save_path):
             try:
                os.makedirs(save_path)
                logger.info(f"Created save directory: {save_path}")
             except OSError as e:
                 logger.error(f"Failed to create save directory {save_path}: {e}")
                 return

        try:
            if self.model_type == 'ultralytics_yolo':
                save_file_path = os.path.join(save_path, os.path.basename(self.path))
                logger.info(f"For YOLO models, 'saving' typically means copying the '.pt' file "
                      f"or exporting. Manual copy or export needed for path '{self.path}'. Target: '{save_file_path}'")

            elif self.model_type == 'VLM':
                 logger.warning(f"Saving logic for model type '{self.model_type}' is not implemented yet.")

            else:
                 logger.warning(f"Saving logic for model type '{self.model_type}' is not implemented yet.")

        except Exception as e:
            logger.error(f"Error saving model {self.model_type} to {save_path}: {e}", exc_info=True)

    def configure(self, config_path):
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file {config_path} does not exist. Using existing settings.")
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            self.priority = config.get('priority', self.priority)
            self.timeout = config.get('timeout', self.timeout)
            logger.info(f"Model {self.model_type} configured with priority {self.priority} and timeout {self.timeout}.")
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from configuration file {config_path}.", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing configuration file {config_path}: {e}", exc_info=True)

    def __str__(self):
        status = "Loaded" if self.model else "Not Loaded"
        return f"Model(type={self.model_type}, path={self.path}, device={self.device}, status={status})"