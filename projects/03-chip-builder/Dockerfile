FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ACCELTWIN_DB_PATH=/data/acceltwin.db \
    ACCELTWIN_EVENT_LOG=/data/events.jsonl

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
RUN mkdir -p /data && useradd --create-home --uid 10001 acceltwin && chown -R acceltwin:acceltwin /data
USER acceltwin
EXPOSE 8080
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]

