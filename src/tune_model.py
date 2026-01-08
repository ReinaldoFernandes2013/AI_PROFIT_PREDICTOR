import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def fine_tune_model():
    print("🧪 Iniciando Laboratório de Otimização (Tuning)...")
    df = pd.read_csv('data/processed_sample.csv')
    
    features = ['price', 'freight_value', 'product_weight_g', 'is_interstate']
    target = 'delivery_time_days'
    
    X = df[features]
    y = df[target]

    # Definindo a grade de parâmetros para testar (Search Grid)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    xgb = XGBRegressor(random_state=42)

    # Configurando a busca exaustiva com Cross-Validation (Visão de QA)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=2,
        n_jobs=-1 # Usa todos os núcleos do seu processador para ser rápido
    )

    print("⏳ Testando as melhores combinações... Isso pode levar um minuto.")
    grid_search.fit(X, y)

    print(f"🏆 Melhor combinação encontrada: {grid_search.best_params_}")
    print(f"📊 Melhor MAE durante o treino: {-grid_search.best_score_:.2f}")

    # Salvando o modelo "Campeão"
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'models/delivery_model.pkl')
    print("💾 Modelo otimizado (Campeão) salvo em models/delivery_model.pkl")

if __name__ == "__main__":
    fine_tune_model()