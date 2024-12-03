# Challenge-4

Execute o script download_data.py para baixar os dados históricos da ação:

cd data
python download_data.py

Treinar o Modelo LSTM

Vá para a pasta models e execute o script lstm_model.py para treinar o modelo:

cd ../models
python lstm_model.py

Executar a API

Vá para a pasta api e execute o script api.py para iniciar a API:

cd ../api
uvicorn api:app --reload

A API estará disponível em http://127.0.0.1:8000. Você pode enviar um POST para /predict/ com os dados históricos e obter previsões de preços.

Visualizar os Dados

Vá para a pasta notebooks e execute o script visualization.py para visualizar os dados históricos:

cd ../notebooks
python visualization.py

Containerizar a API com Docker (Opcional)

Vá para a pasta docker, crie a imagem e execute o contêiner:

cd ../docker
docker build -t stock_prediction_api .
docker run -p 8000:8000 stock_prediction_api

Estrutura do Código

download_data.py

Este script baixa os dados financeiros históricos de uma ação específica usando a biblioteca yfinance e salva os dados em um arquivo CSV.

lstm_model.py

Este script treina um modelo LSTM para prever os preços das ações. Ele utiliza a biblioteca TensorFlow e salva o modelo treinado, além de gerar previsões e salvar os resultados em um arquivo Excel.

api.py

Este script implementa uma API utilizando FastAPI. A API fornece um ponto de acesso para realizar previsões dos preços das ações com base em novos dados históricos fornecidos.

Dockerfile

O Dockerfile containeriza a aplicação FastAPI para facilitar o deploy em qualquer ambiente que suporte Docker.
