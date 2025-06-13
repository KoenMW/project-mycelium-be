from flask import Flask
import os

def check_model_files():
    required_files = [
        "../models/encoder_model.keras",
        "../models/best_hybrid_model.keras"
    ]
    for model_file in required_files:
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

check_model_files()