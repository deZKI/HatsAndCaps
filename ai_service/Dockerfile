FROM python:3.10
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Загрузка модели при сборке образа
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2-cedr-emotion-detection'); \
    AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2-cedr-emotion-detection')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
