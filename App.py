import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import date, timedelta

from DadosBolsa import DadosBolsa
from RegressaoLinear import RegressaoLinear
from RegressaoPolinomial import RegressaoPolinomial
from MediaMovel import MediaMovel
from AvaliadorModelo import AvaliadorModelos


# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="StockPredict",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockPredict")
st.caption(
    "Análise de ações utilizando regressão linear, "
    "regressão polinomial e média móvel."
)
st.divider()


# ==========================================
# EMPRESAS DISPONÍVEIS
# ==========================================

empresas = {
    "Petrobras (PETR4)": "PETR4.SA",
    "Vale (VALE3)": "VALE3.SA",
    "Itaú Unibanco (ITUB4)": "ITUB4.SA",
    "Bradesco (BBDC4)": "BBDC4.SA",
    "Banco do Brasil (BBAS3)": "BBAS3.SA",
    "WEG (WEGE3)": "WEGE3.SA",
    "Ambev (ABEV3)": "ABEV3.SA",
    "Magazine Luiza (MGLU3)": "MGLU3.SA",
    "B3 (B3SA3)": "B3SA3.SA",
    "Localiza (RENT3)": "RENT3.SA"
}


# ==========================================
# CONFIGURAÇÕES DA ANÁLISE
# ==========================================

st.sidebar.header("Configurações")

empresa = st.sidebar.selectbox("Empresa", list(empresas.keys()))
codigo_acao = empresas[empresa]
st.sidebar.caption(f"Código: {codigo_acao}")

st.sidebar.divider()

modo_avaliacao = st.sidebar.radio(
    "Modo de avaliação",
    [
        "Prever período futuro",
        "Testar com os próprios dados de treinamento"
    ],
    help=(
        "No teste com os próprios dados, os modelos são ajustados "
        "e avaliados sobre o mesmo histórico. Isso mede o ajuste aos "
        "dados conhecidos, não a capacidade de prever o futuro."
    )
)

st.sidebar.divider()
st.sidebar.subheader("Período de treinamento")

data_inicio = st.sidebar.date_input(
    "Início dos dados",
    value=date(2024, 1, 1)
)

data_fim = st.sidebar.date_input(
    "Fim dos dados",
    value=date(2024, 12, 31)
)

if modo_avaliacao == "Prever período futuro":
    st.sidebar.divider()
    st.sidebar.subheader("Período de previsão")

    inicio_previsao = st.sidebar.date_input(
        "Início da previsão",
        value=date(2025, 1, 1)
    )

    fim_previsao = st.sidebar.date_input(
        "Fim da previsão",
        value=date(2025, 12, 31)
    )
else:
    inicio_previsao = None
    fim_previsao = None

st.sidebar.divider()
st.sidebar.subheader("Parâmetros dos métodos")

# Sem limite máximo fixo. A validação é feita depois de carregar
# os dados, pois o grau precisa ser menor que a quantidade de amostras.
grau_polinomio = st.sidebar.number_input(
    "Grau do polinômio",
    min_value=1,
    value=2,
    step=1,
    help=(
        "Não há um limite máximo fixo. O grau escolhido, porém, "
        "precisa ser menor que a quantidade de observações de treinamento."
    )
)

periodo_media = st.sidebar.number_input(
    "Período da média móvel",
    min_value=2,
    value=5,
    step=1
)

escala = st.sidebar.selectbox(
    "Escala do gráfico",
    ["Dias", "Semanas", "Meses", "Anos"],
    index=2
)

st.sidebar.divider()

executar = st.sidebar.button(
    "Executar análise",
    type="primary",
    use_container_width=True
)


# ==========================================
# FUNÇÃO PARA DESENHAR O GRÁFICO
# ==========================================

def gerar_grafico(treino, teste, previsoes, escala, modo):
    fig, ax = plt.subplots(figsize=(15, 7))

    if modo == "Prever período futuro":
        ax.plot(
            treino.index,
            treino.values,
            color="gray",
            label="Histórico de treinamento",
            linewidth=1.5
        )

        ax.plot(
            teste.index,
            teste.values,
            color="black",
            label="Valores reais",
            linewidth=2
        )

        ax.axvline(
            teste.index[0],
            color="red",
            linestyle=":",
            label="Início do teste"
        )
    else:
        # Neste modo, treino e teste são o mesmo período.
        ax.plot(
            teste.index,
            teste.values,
            color="black",
            label="Valores reais do treinamento",
            linewidth=2
        )

    cores = {
        "Regressão linear": "blue",
        "Regressão polinomial": "darkorange",
        "Média móvel": "green"
    }

    for nome, valores in previsoes.items():
        ax.plot(
            teste.index,
            valores,
            color=cores[nome],
            linestyle="--",
            linewidth=1.8,
            label=nome
        )

    if escala == "Dias":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    elif escala == "Semanas":
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    elif escala == "Meses":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    elif escala == "Anos":
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    titulo = (
        "Comparação: previsões futuras"
        if modo == "Prever período futuro"
        else "Ajuste dos modelos sobre os dados de treinamento"
    )

    ax.set_title(titulo)
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (R$)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


# ==========================================
# EXECUÇÃO DA ANÁLISE
# ==========================================

if executar:

    try:
        # --------------------------------------
        # 1. VALIDAR DATAS
        # --------------------------------------

        if data_inicio >= data_fim:
            st.error("A data final do treinamento deve ser posterior à inicial.")
            st.stop()

        if modo_avaliacao == "Prever período futuro":
            if inicio_previsao <= data_fim:
                st.error(
                    "O início da previsão deve ser posterior ao fim "
                    "do treinamento."
                )
                st.stop()

            if inicio_previsao > fim_previsao:
                st.error("O início da previsão deve ser anterior ao fim.")
                st.stop()

            data_final_busca = fim_previsao + timedelta(days=1)
        else:
            data_final_busca = data_fim + timedelta(days=1)

        # --------------------------------------
        # 2. BUSCAR DADOS
        # --------------------------------------

        with st.spinner("Buscando dados da bolsa..."):
            dados = DadosBolsa(
                codigo_acao=codigo_acao,
                data_inicio=data_inicio.isoformat(),
                data_fim=data_final_busca.isoformat()
            )

            historico = dados.buscar_dados()

            historico.index = pd.to_datetime(historico.index)
            if historico.index.tz is not None:
                historico.index = historico.index.tz_localize(None)

            historico = historico.sort_index()
            precos = historico["Close"].dropna()

        # --------------------------------------
        # 3. SEPARAR TREINAMENTO E TESTE
        # --------------------------------------

        treino = precos[
            (precos.index >= pd.Timestamp(data_inicio)) &
            (precos.index < pd.Timestamp(data_fim + timedelta(days=1)))
        ]

        if treino.empty:
            st.error("Não foram encontrados dados no período de treinamento.")
            st.stop()

        if len(treino) <= grau_polinomio:
            st.error(
                f"O grau {grau_polinomio} exige pelo menos "
                f"{grau_polinomio + 1} observações. "
                f"O período selecionado possui {len(treino)}."
            )
            st.stop()

        if len(treino) < periodo_media:
            st.error(
                f"O período da média móvel ({periodo_media}) não pode ser "
                f"maior que a quantidade de dados de treinamento ({len(treino)})."
            )
            st.stop()

        if modo_avaliacao == "Testar com os próprios dados de treinamento":
            # Para comparar todos os métodos sobre as mesmas datas,
            # começamos no primeiro ponto em que existe média móvel.
            inicio_validacao = periodo_media - 1
            teste = treino.iloc[inicio_validacao:]

            indices_teste = np.arange(
                inicio_validacao,
                len(treino)
            )
        else:
            dados_pos_treino = precos[
                (precos.index > treino.index[-1]) &
                (precos.index <= pd.Timestamp(fim_previsao))
            ]

            mascara_teste = (
                (dados_pos_treino.index >= pd.Timestamp(inicio_previsao)) &
                (dados_pos_treino.index < pd.Timestamp(fim_previsao + timedelta(days=1)))
            )

            teste = dados_pos_treino[mascara_teste]

            if len(dados_pos_treino) == 0:
                st.error("Não existem pregões após o treinamento.")
                st.stop()

            if teste.empty:
                st.error("Não foram encontrados dados no período de previsão.")
                st.stop()

            indices_teste = np.arange(
                len(treino),
                len(treino) + len(dados_pos_treino)
            )[mascara_teste]

        st.success("Dados carregados com sucesso!")

        # --------------------------------------
        # 4. TREINAR MODELOS
        # --------------------------------------

        with st.spinner("Calculando modelos..."):
            valores_treino = treino.values

            linear = RegressaoLinear()
            linear.ajustar(valores_treino)

            polinomial = RegressaoPolinomial(grau=int(grau_polinomio))
            polinomial.ajustar(valores_treino)

            media = MediaMovel(periodo=int(periodo_media))
            media.ajustar(valores_treino)

            # ----------------------------------
            # 5. GERAR PREVISÕES
            # ----------------------------------

            if modo_avaliacao == "Testar com os próprios dados de treinamento":
                # Regressões são avaliadas nos mesmos índices usados no ajuste.
                previsao_linear = np.asarray(
                    linear.prever(indices_teste)
                )

                previsao_polinomial = np.asarray(
                    polinomial.prever(indices_teste)
                )

                # A média móvel calculada no ajuste possui len(treino)-periodo+1
                # valores e começa na observação de índice periodo-1.
                previsao_media = np.convolve(
                    valores_treino,
                    np.ones(int(periodo_media)) / int(periodo_media),
                    mode="valid"
                )

            else:
                quantidade_pos_treino = len(dados_pos_treino)

                indices_completos = np.arange(
                    len(treino),
                    len(treino) + quantidade_pos_treino
                )

                previsao_linear_completa = np.asarray(
                    linear.prever(indices_completos)
                )

                previsao_polinomial_completa = np.asarray(
                    polinomial.prever(indices_completos)
                )

                previsao_media_completa = np.asarray(
                    media.prever(quantidade_pos_treino)
                )

                previsao_linear = previsao_linear_completa[mascara_teste]
                previsao_polinomial = previsao_polinomial_completa[mascara_teste]
                previsao_media = previsao_media_completa[mascara_teste]

            previsoes = {
                "Regressão linear": previsao_linear,
                "Regressão polinomial": previsao_polinomial,
                "Média móvel": previsao_media
            }

        # --------------------------------------
        # 6. AVALIAR MODELOS
        # --------------------------------------

        avaliador = AvaliadorModelos
        resultados = {}

        for nome, previsao in previsoes.items():
            resultados[nome] = avaliador.calcular_metricas(
                teste.values,
                previsao
            )

        # --------------------------------------
        # 7. APRESENTAR INFORMAÇÕES
        # --------------------------------------

        st.divider()
        st.header(f"Resultados: {empresa}")

        if modo_avaliacao == "Testar com os próprios dados de treinamento":
            st.info(
                "Teste interno: os modelos foram ajustados e avaliados "
                "com o mesmo conjunto de dados. Os resultados mostram "
                "o quanto cada método se ajusta ao histórico, mas não "
                "representam uma previsão de dados desconhecidos."
            )

        col1, col2, col3 = st.columns(3)

        col1.metric("Dias de treinamento", len(treino))
        col2.metric("Dias avaliados", len(teste))
        col3.metric("Preço real final", f"R$ {teste.iloc[-1]:.2f}")

        # --------------------------------------
        # 8. GRÁFICO
        # --------------------------------------

        st.subheader("Gráfico comparativo")

        fig = gerar_grafico(
            treino=treino,
            teste=teste,
            previsoes=previsoes,
            escala=escala,
            modo=modo_avaliacao
        )

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------
        # 9. TABELA DE MÉTRICAS
        # --------------------------------------

        st.divider()
        st.subheader("Avaliação dos modelos")

        tabela = pd.DataFrame(resultados).T
        tabela.index.name = "Modelo"

        st.dataframe(
            tabela.style.format("{:.4f}"),
            use_container_width=True
        )

        # --------------------------------------
        # 10. MELHOR MODELO
        # --------------------------------------

        melhor_modelo = tabela["MAE"].idxmin()
        st.success(f"Menor erro absoluto médio: {melhor_modelo}")

        # --------------------------------------
        # 11. EXPLICAÇÃO
        # --------------------------------------

        with st.expander("Sobre os resultados"):
            st.write("""
            **MAE:** representa o erro absoluto médio entre os preços reais
            e previstos.

            **RMSE:** penaliza mais fortemente os erros grandes.

            **R²:** representa o quanto das variações dos preços reais é
            explicado pelas previsões.

            O menor MAE indica o menor erro médio absoluto nesta avaliação.
            Isso não garante que o modelo será o melhor em períodos futuros.
            """)

            if modo_avaliacao == "Testar com os próprios dados de treinamento":
                st.warning(
                    "Como os mesmos dados foram usados para ajustar e avaliar "
                    "os modelos, as métricas podem parecer melhores do que "
                    "seriam em dados novos. Esse modo é útil para visualizar "
                    "o ajuste matemático dos métodos."
                )

            if grau_polinomio >= 6:
                st.warning(
                    "Graus polinomiais elevados podem causar sobreajuste "
                    "e oscilações, principalmente ao extrapolar para datas "
                    "futuras. Não há limite fixo na interface, mas isso "
                    "não significa que graus muito altos sejam adequados."
                )

        # --------------------------------------
        # 12. EXPORTAR RESULTADOS
        # --------------------------------------

        dados_exportacao = pd.DataFrame({
            "Data": teste.index,
            "Preço real": teste.values,
            "Regressão linear": previsao_linear,
            "Regressão polinomial": previsao_polinomial,
            "Média móvel": previsao_media
        })

        csv = dados_exportacao.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Baixar resultados CSV",
            data=csv,
            file_name=f"{codigo_acao}_resultados.csv",
            mime="text/csv"
        )

    except Exception as erro:
        st.error(f"Não foi possível executar a análise: {erro}")
        st.exception(erro)

else:
    st.info(
        "Configure a empresa, os períodos e os parâmetros na barra lateral. "
        "Depois clique em 'Executar análise'."
    )

    st.markdown("""
    ### Como funciona?

    1. Escolha uma ação da bolsa.
    2. Defina o período de treinamento.
    3. Escolha se deseja prever um período futuro ou testar o ajuste
       usando os próprios dados de treinamento.
    4. Configure o grau polinomial e o período da média móvel.
    5. Execute a análise e compare os resultados.
    """)
