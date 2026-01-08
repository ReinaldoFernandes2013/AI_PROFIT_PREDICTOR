import streamlit as st
import joblib
import pandas as pd

# Configuração de Qualidade da Página
st.set_page_config(page_title="AI Logistics Optimizer", page_icon="🚚")

st.title("🚚 AI Logistics Optimizer")
st.subheader("Previsão Inteligente de Tempo de Entrega")

# Carregando o cérebro da nossa IA (AI Scientist)
@st.cache_resource
def load_model():
    return joblib.load('models/delivery_model.pkl')

model = load_model()

# Interface de Entrada de Dados
st.sidebar.header("Parâmetros do Produto")
price = st.sidebar.number_input("Preço do Produto (R$)", min_value=0.0, value=100.0)
freight = st.sidebar.number_input("Valor do Frete (R$)", min_value=0.0, value=20.0)
weight = st.sidebar.number_input("Peso (gramas)", min_value=0, value=1000)

# Botão de Predição
if st.button("Calcular Prazo Estimado"):
    # Criando o DataFrame para o modelo (ML Engineer)
    input_data = pd.DataFrame([[price, freight, weight]], 
                              columns=['price', 'freight_value', 'product_weight_g'])
    
    prediction = model.predict(input_data)[0]
    
    # Exibição do Resultado de QA
    st.success(f"📦 A previsão de entrega para este pedido é de: **{prediction:.1f} dias**")
    
    # Alerta de Insight
    if prediction > 15:
        st.warning("⚠️ Atenção: Prazo de entrega acima da média da Olist.")