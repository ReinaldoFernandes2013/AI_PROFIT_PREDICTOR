# 🚚 AI Logistics Optimizer - Previsão de Entrega com Regressão

Este projeto utiliza Inteligência Artificial para prever o tempo de entrega de mercadorias no e-commerce (Dataset Olist), ajudando na otimização logística e na transparência com o cliente.

## 🧠 Perfil do Projeto
- **Papel**: AI Scientist & Machine Learning Engineer.
- **Problema**: Regressão (Previsão de valores contínuos/dias).
- **Modelo**: XGBoost Regressor.
- **Interface**: Dashboard Interativo com Streamlit.

## 🛠️ Tecnologias e Bibliotecas
- **Python 3.12**
- **Pandas & Numpy**: Processamento de dados.
- **Scikit-Learn**: Divisão de treino/teste e métricas (MAE, R²).
- **XGBoost**: Algoritmo de alta performance para regressão.
- **Streamlit**: Interface Web de predição.

## 📊 Performance do Modelo
O modelo foi treinado com mais de 88.000 amostras e alcançou:
- **Erro Médio Absoluto (MAE)**: ~5.44 dias.
- **Variáveis Principais**: Preço, Valor do Frete e Peso do Produto.

## 📂 Estrutura de Pastas
- `data/`: Dados brutos e processados.
- `models/`: Arquivos binários do modelo treinado (`.pkl`).
- `src/`: Scripts de engenharia de dados, treinamento e App.
- `notebooks/`: Análises exploratórias e visualizações.

## 🚀 Como Executar
1. Ative o ambiente virtual: `source venv/Scripts/activate`
2. Instale as dependências: `pip install -r requirements.txt`
3. Rode o App: `streamlit run src/app.py`