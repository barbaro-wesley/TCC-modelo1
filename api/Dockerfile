FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt && useradd --uid 10001 --create-home api
COPY api /app/api
USER api
EXPOSE 8080
CMD ["uvicorn", "api.app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-proxy-headers"]
