FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY router/ router/
COPY retrievers/ retrievers/
COPY ingestion/ ingestion/
COPY utils/ utils/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
