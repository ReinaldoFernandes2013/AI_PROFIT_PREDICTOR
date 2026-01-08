import pytest
import pandas as pd
import os

def test_data_loading():
    """Verifica se os arquivos CSV essenciais existem na pasta data"""
    data_path = 'data/'
    required_files = ['olist_orders_dataset.csv', 'olist_order_items_dataset.csv']
    
    for file in required_files:
        assert os.path.exists(os.path.join(data_path, file)), f"❌ Arquivo {file} não encontrado!"

def test_processed_sample_exists():
    """Verifica se o script de engenharia de dados gerou a amostra processada"""
    assert os.path.exists('data/processed_sample.csv'), "❌ O arquivo processado não foi gerado!"

def test_model_loading():
    """Verifica se o modelo treinado (.pkl) está presente na pasta models"""
    assert os.path.exists('models/delivery_model.pkl'), "❌ O modelo treinado não foi encontrado!"