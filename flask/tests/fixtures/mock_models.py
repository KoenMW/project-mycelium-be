mock_models.py
from unittest.mock import MagicMock

class MockEncoderModel:
    def predict(self, input_data):
        return "mock_prediction"

class MockBestHybridModel:
    def predict(self, input_data):
        return "mock_hybrid_prediction"

encoder_model = MockEncoderModel()
best_hybrid_model = MockBestHybridModel()