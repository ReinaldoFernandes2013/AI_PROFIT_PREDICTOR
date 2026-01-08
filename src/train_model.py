import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_delivery_predictor():
    print("🚀 Carregando dados refinados para treinamento...")
    # Lendo o arquivo que acabamos de gerar com a feature is_interstate
    df = pd.read_csv('data/processed_sample.csv')
    
    # 🔍 AI Scientist: Agora com inteligência de localização
    # Adicionamos 'is_interstate' para ajudar o modelo a entender prazos logísticos longos
    features = ['price', 'freight_value', 'product_weight_g', 'is_interstate']
    target = 'delivery_time_days'
    
    # Limpeza de QA: Garantir que não existam nulos nas novas features
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    # Divisão treino/teste (Padrão Ouro de ML Engineer)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"📈 Treinando XGBoost Regressor Refinado com {len(X_train)} amostras...")
    # Mantendo os hiperparâmetros iniciais para comparar a melhora apenas da feature
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    
    # Avaliação de Qualidade (QA)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"✅ Treinamento concluído com sucesso!")
    print(f"📊 Novo Erro Médio Absoluto (MAE): {mae:.2f} dias")
    print(f"📊 Novo Coeficiente R²: {r2:.2f}")
    
    # Salvando o modelo atualizado
    joblib.dump(model, 'models/delivery_model.pkl')
    print("💾 Modelo atualizado salvo em models/delivery_model.pkl")

if __name__ == "__main__":
    train_delivery_predictor()