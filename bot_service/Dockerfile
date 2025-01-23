FROM python:3.10

WORKDIR /bot_service

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app/main.py"]