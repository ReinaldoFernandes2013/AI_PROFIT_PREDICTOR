import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando a amostra que geramos
df = pd.read_csv('data/processed_sample.csv')

# Configuração de Qualidade Visual
plt.figure(figsize=(10, 6))
sns.histplot(df['delivery_time_days'].dropna(), bins=50, kde=True, color='blue')
plt.title('Distribuição do Tempo de Entrega (Dias) - Visão AI Scientist')
plt.xlabel('Dias para Entrega')
plt.ylabel('Frequência de Pedidos')

# Salvando o insight
plt.savefig('notebooks/distribuicao_entrega.png')
print("📊 Gráfico de análise logística gerado em notebooks/distribuicao_entrega.png")