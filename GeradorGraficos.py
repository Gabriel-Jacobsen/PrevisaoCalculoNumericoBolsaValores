
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


class GeradorGraficos:

    def __init__(self, titulo="Análise de preços de ações"):
        self.titulo = titulo

    def gerar_grafico(
        self,
        datas,
        valores_reais,
        previsao_linear=None,
        previsao_polinomial=None,
        previsao_media_movel=None,

        data_inicio=None,
        data_fim=None,

        inicio_previsao=None,
        fim_previsao=None,

        escala_tempo="meses",
        limite_min=None,
        limite_max=None,

        salvar=False,
        nome_arquivo="grafico_stockpredict.png"
    ):

        # ==========================================
        # 1. PREPARAÇÃO DOS DADOS
        # ==========================================

        datas = pd.DatetimeIndex(pd.to_datetime(datas))

        # Remove fuso horário, caso exista
        if datas.tz is not None:
            datas = datas.tz_localize(None)

        valores_reais = np.asarray(
            valores_reais, dtype=float
        )

        if len(datas) != len(valores_reais):
            raise ValueError(
                "Datas e valores reais possuem tamanhos diferentes."
            )

        # Organiza os métodos de previsão
        previsoes = {
            "Regressão linear": previsao_linear,
            "Regressão polinomial": previsao_polinomial,
            "Média móvel": previsao_media_movel
        }

        for nome, previsao in previsoes.items():

            if previsao is not None:
                previsoes[nome] = np.asarray(
                    previsao, dtype=float
                )

                if len(previsoes[nome]) != len(datas):
                    raise ValueError(
                        f"A previsão de {nome} deve possuir "
                        "o mesmo tamanho do histórico."
                    )

        # ==========================================
        # 2. DEFINIÇÃO DOS INTERVALOS
        # ==========================================

        if not datas.is_monotonic_increasing:
            raise ValueError("As datas precisam estar em ordem crescente.")

        data_inicio = (
            pd.Timestamp(data_inicio)
            if data_inicio is not None
            else datas.min()
        )

        data_fim = (
            pd.Timestamp(data_fim)
            if data_fim is not None
            else datas.max()
        )

        inicio_previsao = (
            pd.Timestamp(inicio_previsao)
            if inicio_previsao is not None
            else data_inicio
        )

        fim_previsao = (
            pd.Timestamp(fim_previsao)
            if fim_previsao is not None
            else data_fim
        )

        if data_inicio > data_fim:
            raise ValueError("A data inicial deve ser anterior à final.")

        if inicio_previsao > fim_previsao:
            raise ValueError(
                "O início da previsão deve ser anterior ao fim."
            )

        if inicio_previsao < data_inicio or fim_previsao > data_fim:
            raise ValueError(
                "O intervalo de previsão deve estar dentro "
                "do período selecionado."
            )

        # Máscaras para filtrar as datas
        mascara_historico = (
            (datas >= data_inicio) &
            (datas <= data_fim)
        )

        mascara_previsao = (
            (datas >= inicio_previsao) &
            (datas <= fim_previsao)
        )

        if not mascara_historico.any():
            raise ValueError("Não existem dados no período selecionado.")

        if not mascara_previsao.any():
            raise ValueError("Não existem dados no período de previsão.")

        datas_filtradas = datas[mascara_historico]
        valores_filtrados = valores_reais[mascara_historico]

        # ==========================================
        # 3. CRIAÇÃO DO GRÁFICO
        # ==========================================

        fig, ax = plt.subplots(figsize=(16, 8))

        # Valores reais
        ax.plot(
            datas_filtradas,
            valores_filtrados,
            label="Valores reais",
            color="black",
            linewidth=2
        )

        # ==========================================
        # 4. DESENHO DAS PREVISÕES
        # ==========================================

        cores = {
            "Regressão linear": "blue",
            "Regressão polinomial": "darkorange",
            "Média móvel": "green"
        }

        for nome, previsao in previsoes.items():

            if previsao is None:
                continue

            ax.plot(
                datas[mascara_previsao],
                previsao[mascara_previsao],
                label=nome,
                color=cores[nome],
                linestyle="--",
                linewidth=1.8
            )

        # Início e fim da previsão
        ax.axvline(
            inicio_previsao,
            color="red",
            linestyle=":",
            linewidth=1.5,
            label="Início da previsão"
        )

        ax.axvline(
            fim_previsao,
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label="Fim da previsão"
        )

        # ==========================================
        # 5. ESCALA TEMPORAL
        # ==========================================

        if escala_tempo == "dias":

            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=5)
            )

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%d/%m/%Y")
            )

        elif escala_tempo == "semanas":

            ax.xaxis.set_major_locator(
                mdates.WeekdayLocator(interval=1)
            )

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%d/%m/%Y")
            )

        elif escala_tempo == "meses":

            ax.xaxis.set_major_locator(
                mdates.MonthLocator(interval=1)
            )

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%m/%Y")
            )

        elif escala_tempo == "anos":

            ax.xaxis.set_major_locator(
                mdates.YearLocator()
            )

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%Y")
            )

        else:
            raise ValueError(
                "Escala inválida: dias, semanas, meses ou anos."
            )

        # ==========================================
        # 6. ESCALA DOS PREÇOS
        # ==========================================

        if limite_min is not None or limite_max is not None:

            ax.set_ylim(
                bottom=limite_min,
                top=limite_max
            )

        # ==========================================
        # 7. CONFIGURAÇÕES VISUAIS
        # ==========================================

        ax.set_title(self.titulo)
        ax.set_xlabel("Data")
        ax.set_ylabel("Preço da ação (R$)")

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if salvar:
            plt.savefig(nome_arquivo, dpi=300)

        plt.show()
