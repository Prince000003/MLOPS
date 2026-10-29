FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas scikit-learn joblib

COPY src/ src/
COPY models/ models/

# Runs a sample prediction with the trained model
CMD ["python", "src/predict.py"]
