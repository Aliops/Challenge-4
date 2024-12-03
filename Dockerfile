# Dockerfile para containerização da API FastAPI
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Copiar os diretórios necessários para dentro do container
COPY ./api /app/api
COPY ./models /app/models
COPY ./data /app/data
COPY ./utils /app/utils

# Definir o diretório de trabalho
WORKDIR /app

# Copiar o arquivo de dependências
COPY requirements.txt /app

# Instalar as dependências necessárias
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta que o Uvicorn utilizará
EXPOSE 8000

# Comando padrão para iniciar o servidor
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
