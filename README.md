# 🚚 AI Logistics Optimizer - Previsão de Entrega com Regressão (v3.0 - Final)

Este projeto utiliza Inteligência Artificial de ponta para prever o tempo de entrega de mercadorias no e-commerce (Dataset Olist). Através de ciclos de refinamento e otimização de hiperparâmetros, entregamos uma solução robusta para suporte à decisão logística.

## 🧠 Perfil do Projeto
- **Papéis**: AI Scientist, Machine Learning Engineer & Quality Assurance (QA).
- **Tipo de Problema**: Regressão (Previsão de dias de entrega).
- **Modelo Final**: XGBoost Regressor Otimizado via GridSearchCV.
- **Interface**: Dashboard Interativo v2 com inteligência geográfica.

## 🛠️ Tecnologias e Bibliotecas
- **Python 3.12**
- **Pandas & Numpy**: Processamento de dados e Feature Engineering.
- **XGBoost**: Algoritmo de alta performance para regressão.
- **Scikit-Learn**: Tuning de hiperparâmetros e métricas de validação.
- **Streamlit**: Deploy de interface web intuitiva.
- **Pytest**: Automação de testes de qualidade e integridade.

## 📊 Performance e Otimização Final
O projeto evoluiu através de três estágios de maturidade, atingindo o estado da arte com o Tuning de Hiperparâmetros.

| Métrica | Versão 1.0 (Base) | Versão 2.0 (Refinada) | Versão 3.0 (Tuning) |
| :--- | :--- | :--- | :--- |
| **Erro Médio (MAE)** | 5.44 dias | 5.11 dias | **5.07 dias** ✅ |
| **Coeficiente R²** | 0.17 | **0.23** | **Otimizado** 📈 |
| **Configuração** | Padrão | Feature Geográfica | **GridSearchCV** 🏆 |

### 🏆 Configuração Campeã (XGBoost):
- `learning_rate`: 0.05 | `max_depth`: 8 | `n_estimators`: 200 | `subsample`: 0.8

## 🧪 Garantia de Qualidade (QA)
Para garantir a confiabilidade do software em produção, implementamos testes automatizados que validam:
- **Pipeline de Dados**: Existência e integridade dos arquivos CSV.
- **Processamento**: Verificação da geração do dataset limpo e filtrado (0-60 dias).
- **Persistência do Modelo**: Garantia de que o arquivo `.pkl` otimizado está pronto para o deploy.

Para rodar os testes:
```bash
pytest

📂 Estrutura de Pastas
data/: Datasets originais e processados.

models/: Binário do modelo campeão (delivery_model.pkl).

src/: Scripts de preparação, treinamento, tuning e App Streamlit.

tests/: Scripts de testes automatizados com Pytest.

notebooks/: Análises exploratórias e visualizações logísticas.

🚀 Como Executar
Ative o ambiente virtual: source venv/Scripts/activate

Instale as dependências: pip install -r requirements.txt

Execute o App: streamlit run src/app.py