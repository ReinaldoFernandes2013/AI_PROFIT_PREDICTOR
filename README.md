# 🚚 AI Logistics Optimizer - Previsão de Entrega com Regressão (v2.0)

Este projeto utiliza Inteligência Artificial de ponta para prever o tempo de entrega de mercadorias no e-commerce (Dataset Olist), auxiliando na transparência logística e na tomada de decisão estratégica.

## 🧠 Perfil do Projeto
- **Papéis**: AI Scientist, Machine Learning Engineer & Quality Assurance (QA).
- **Tipo de Problema**: Regressão (Previsão de dias de entrega).
- **Modelo Principal**: XGBoost Regressor.
- **Interface**: Dashboard Interativo v2 com inteligência geográfica.

## 🛠️ Tecnologias e Bibliotecas
- **Python 3.12**
- **Pandas & Numpy**: Processamento e engenharia de atributos.
- **XGBoost**: Algoritmo de alta performance para regressão.
- **Streamlit**: Deploy de interface web intuitiva.
- **Pytest**: Automação de testes de qualidade.

## 📊 Performance e Refinamento
O projeto passou por um ciclo de otimização focado em **Engenharia de Atributos** e limpeza de dados.

| Métrica | Versão Inicial | Versão 2.0 (Refinada) | Status |
| :--- | :--- | :--- | :--- |
| **Erro Médio (MAE)** | 5.44 dias | **5.11 dias** | ✅ Melhoria na Precisão |
| **Coeficiente R²** | 0.17 | **0.23** | 📈 Ganho de 35% de performance |
| **Principais Atributos** | Preço, Frete, Peso | **+ Localização Interestadual** | 🌎 Inteligência Geográfica |

## 🧪 Garantia de Qualidade (QA)
Para garantir a confiabilidade do software em produção, implementamos testes automatizados que validam:
- **Pipeline de Dados**: Existência e integridade dos arquivos CSV.
- **Processamento**: Verificação da geração do dataset limpo e filtrado.
- **Persistência do Modelo**: Garantia de que o arquivo `.pkl` está pronto para o deploy.

Para rodar os testes:
```bash
pytest

📂 Estrutura de Pastas
data/: Datasets originais e processados (amostras filtradas de 0 a 60 dias).

models/: Binários do modelo treinado (delivery_model.pkl).

src/: Scripts de preparação de dados, treinamento e App Streamlit.

tests/: Scripts de testes automatizados com Pytest.

notebooks/: Análises exploratórias e visualizações logísticas.

🚀 Como Executar
Ative o ambiente virtual: source venv/Scripts/activate

Instale as dependências: pip install -r requirements.txt

Execute o App: streamlit run src/app.py

