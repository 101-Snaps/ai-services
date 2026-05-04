FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirments.txt

COPY app.py .
COPY model.pt .

ENV AI_PORT=5000
EXPOSE 5000

CMD ["gunicorn", "app:app", "--timeout", "120", "--workers", "1"]
