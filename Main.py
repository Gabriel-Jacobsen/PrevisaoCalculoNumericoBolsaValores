
from DadosBolsa import DadosBolsa
from RegressaoLinear import RegressaoLinear
from RegressaoPolinomial import RegressaoPolinomial
from MediaMovel import MediaMovel
from GeradorGraficos import GeradorGraficos
import numpy as np
import pandas as pd


# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================

codigo_acao = "PETR4.SA"

# ==========================================
# 2. BUSCAR DADOS DE 2024
# ==========================================

dados_2024 = DadosBolsa(
    codigo_acao=codigo_acao,
    data_inicio="2024-01-01",
    data_fim="2024-10-01"
)

historico_2024 = dados_2024.buscar_dados()

datas_2024 = historico_2024.index
precos_2024 = historico_2024["Close"].values


# ==========================================
# 3. BUSCAR DADOS REAIS DE 2025
# ==========================================

dados_2025 = DadosBolsa(
    codigo_acao=codigo_acao,
    data_inicio="2024-01-01",
    data_fim="2024-10-01"
)

historico_2025 = dados_2025.buscar_dados()

datas_2025 = historico_2025.index
precos_reais_2025 = historico_2025["Close"].values


# ==========================================
# 4. TREINAR OS MÉTODOS COM 2024
# ==========================================

linear = RegressaoLinear()
linear.ajustar(precos_2024)

polinomial = RegressaoPolinomial(grau=30)
polinomial.ajustar(precos_2024)

# ==========================================
# 5. REALIZAR PREVISÕES PARA 2025
# ==========================================

quantidade_dias = len(precos_2024)

# Índices posteriores ao último dia de 2024
# indices_2025 = np.arange(
#     len(precos_2024),
#     len(precos_2024) + quantidade_dias
# )

# indices_2024 = len(precos_2024)
indices = np.arange(len(precos_2024))

previsao_linear = linear.prever(indices)

previsao_polinomial = polinomial.prever(indices)

media = MediaMovel(periodo=quantidade_dias)
media.ajustar(precos_2024)

# previsao_linear = linear.prever(indices_2025)

# previsao_polinomial = polinomial.prever(indices_2025)

previsao_media_movel = media.prever(quantidade_dias)


# ==========================================
# 6. GERAR GRÁFICO COMPARATIVO
# ==========================================


gerador = GeradorGraficos(
    titulo="PETR4 - Análise de regressão"
)

gerador.gerar_grafico(

    datas=datas_2024,

    valores_reais=precos_2024,

    previsao_linear=previsao_linear,

    previsao_polinomial=previsao_polinomial,

    previsao_media_movel=previsao_media_movel,

    # Período total exibido
    data_inicio="2024-01-01",
    data_fim="2024-12-31",

    # Período destacado pelas previsões
    inicio_previsao="2024-07-01",
    fim_previsao="2024-09-30",

    escala_tempo="meses",

    salvar=True
)

