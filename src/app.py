import streamlit as st
import joblib
import pandas as pd

# Configuração de Qualidade da Página
st.set_page_config(page_title="AI Logistics Optimizer v2", page_icon="🚚")

st.title("🚚 AI Logistics Optimizer - Versão Refinada")
st.subheader("Previsão Inteligente com Inteligência Geográfica")

# Carregando o modelo atualizado (AI Scientist)
@st.cache_resource
def load_model():
    return joblib.load('models/delivery_model.pkl')

model = load_model()

# Interface de Entrada de Dados (ML Engineer)
st.sidebar.header("Parâmetros do Produto")
price = st.sidebar.number_input("Preço do Produto (R$)", min_value=0.0, value=100.0)
freight = st.sidebar.number_input("Valor do Frete (R$)", min_value=0.0, value=20.0)
weight = st.sidebar.number_input("Peso (gramas)", min_value=0, value=1000)

# Nova Feature: Localização (QA)
interstate_option = st.sidebar.selectbox(
    "A entrega é para outro estado?",
    options=["Não (Mesmo Estado)", "Sim (Outro Estado)"]
)
# Convertendo para o formato que a IA entende (0 ou 1)
is_interstate = 1 if interstate_option == "Sim (Outro Estado)" else 0

# Botão de Predição
if st.button("Calcular Prazo Estimado"):
    # Criando o DataFrame com a nova ordem de features
    input_data = pd.DataFrame(
        [[price, freight, weight, is_interstate]], 
        columns=['price', 'freight_value', 'product_weight_g', 'is_interstate']
    )
    
    prediction = model.predict(input_data)[0]
    
    # Exibição do Resultado de QA
    st.success(f"📦 A nova previsão estimada é de: **{prediction:.1f} dias**")
    
    # Insights de Especialista
    if is_interstate == 1:
        st.info("ℹ️ O modelo considerou o acréscimo logístico de transporte interestadual.")