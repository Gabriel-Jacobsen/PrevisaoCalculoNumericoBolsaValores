
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class GeradorGraficos:

    def __init__(self, titulo="Previsão de preços de ações"):
        self.titulo = titulo

    def gerar_grafico(
        self,
        datas,
        valores_reais,
        previsao_linear=None,
        previsao_polinomial=None,
        previsao_media_movel=None,
        datas_previsao=None,
        escala_tempo="dias",
        salvar=False,
        nome_arquivo="grafico_stockpredict.png"
    ):

        # Converte os dados para os formatos necessários
        datas = pd.to_datetime(datas)
        valores_reais = np.array(valores_reais, dtype=float)

        # Cria o gráfico
        plt.figure(figsize=(14, 7))

        # Valores reais
        plt.plot(
            datas,
            valores_reais,
            label="Valores reais",
            color="black",
            linewidth=2
        )

        # Regressão linear
        if previsao_linear is not None:
            plt.plot(
                datas_previsao,
                previsao_linear,
                label="Regressão linear",
                linestyle="--"
            )

        # Regressão polinomial
        if previsao_polinomial is not None:
            plt.plot(
                datas_previsao,
                previsao_polinomial,
                label="Regressão polinomial",
                linestyle="--"
            )

        # Média móvel
        if previsao_media_movel is not None:
            plt.plot(
                datas_previsao,
                previsao_media_movel,
                label="Média móvel",
                linestyle="--"
            )

        # Configura a escala de tempo
        eixo = plt.gca()

        if escala_tempo == "dias":
            eixo.xaxis.set_major_locator(
                plt.matplotlib.dates.DayLocator(interval=5)
            )

        elif escala_tempo == "semanas":
            eixo.xaxis.set_major_locator(
                plt.matplotlib.dates.WeekdayLocator(interval=2)
            )

        elif escala_tempo == "meses":
            eixo.xaxis.set_major_locator(
                plt.matplotlib.dates.MonthLocator(interval=1)
            )

        elif escala_tempo == "anos":
            eixo.xaxis.set_major_locator(
                plt.matplotlib.dates.YearLocator()
            )

        else:
            raise ValueError(
                "Escala inválida. Use dias, semanas, meses ou anos."
            )

        # Formatação das datas
        eixo.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%d/%m/%Y")
        )

        plt.title(self.titulo)
        plt.xlabel("Data")
        plt.ylabel("Preço da ação (R$)")

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Salva o gráfico, caso solicitado
        if salvar:
            plt.savefig(nome_arquivo, dpi=300)

        plt.show()
