import pandas as pd
import os

def load_and_merge_data():
    # Caminho para os dados
    data_path = 'data/'
    
    print("🚀 Iniciando carregamento de dados...")
    
    # Lendo os datasets principais
    orders = pd.read_csv(os.path.join(data_path, 'olist_orders_dataset.csv'))
    items = pd.read_csv(os.path.join(data_path, 'olist_order_items_dataset.csv'))
    products = pd.read_csv(os.path.join(data_path, 'olist_products_dataset.csv'))
    
    # Merge de Engenharia de Dados
    df = pd.merge(items, orders, on='order_id')
    df = pd.merge(df, products, on='product_id')
    
    # Criando colunas para nossa Regressão (AI Scientist)
    # Exemplo: Tempo de entrega real em dias
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    
    print(f"✅ Fusão completa! Shape do dataset: {df.shape}")
    return df

if __name__ == "__main__":
    dataset = load_and_merge_data()
    # Salvando uma amostra para inspeção de QA
    dataset.to_csv('data/processed_sample.csv', index=False)
    print("💾 Amostra processada salva em data/processed_sample.csv")