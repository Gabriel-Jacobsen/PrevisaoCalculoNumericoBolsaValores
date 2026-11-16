
from DadosBolsa import DadosBolsa
from RegressaoLinear import RegressaoLinear
from RegressaoPolinomial import RegressaoPolinomial
from MediaMovel import MediaMovel
from GeradorGraficos import GeradorGraficos
import numpy as np
import pandas as pd

# 1. Buscar os dados
dados = DadosBolsa(
    codigo_acao="PETR4.SA",
    data_inicio="2024-01-01",
    data_fim="2024-12-01",
    intervalo="1d"
)

historico = dados.buscar_dados()

datas = historico.index
precos = historico["Close"].values

# 2. Treinar os métodos
linear = RegressaoLinear()
linear.ajustar(precos)

polinomial = RegressaoPolinomial(grau=2)
polinomial.ajustar(precos)

media = MediaMovel(periodo=30)
media.ajustar(precos)

# 3. Gerar previsões
quantidade_dias = 10

indices_futuros = np.arange(
    len(precos),
    len(precos) + quantidade_dias
)

previsao_linear = linear.prever(indices_futuros)
previsao_polinomial = polinomial.prever(indices_futuros)
previsao_media = media.prever(quantidade_dias)

# 4. Criar datas futuras
datas_futuras = pd.bdate_range(
    start=datas[-1] + pd.Timedelta(days=1),
    periods=quantidade_dias
)

# 5. Gerar gráfico
gerador = GeradorGraficos(
    titulo="Análise de previsão - PETR4"
)

gerador.gerar_grafico(
    datas=datas,
    valores_reais=precos,
    previsao_linear=previsao_linear,
    previsao_polinomial=previsao_polinomial,
    previsao_media_movel=previsao_media,
    datas_previsao=datas_futuras,
    escala_tempo="meses",
    salvar=True
)
