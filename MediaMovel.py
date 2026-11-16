
import numpy as np
import matplotlib.pyplot as plt


class MediaMovel:

    def __init__(self, periodo=5):

        if periodo < 1:
            raise ValueError("O período deve ser maior que zero.")

        self.periodo = periodo
        self.medias = None
        self.precos = None

    def ajustar(self, precos):

        self.precos = np.array(precos, dtype=float)

        if len(self.precos) < self.periodo:
            raise ValueError(
                "Quantidade de dados insuficiente para o período."
            )

        if not np.all(np.isfinite(self.precos)):
            raise ValueError("Os preços contêm valores inválidos.")

        # Calcular médias móveis
        self.medias = np.convolve(
            self.precos,
            np.ones(self.periodo) / self.periodo,
            mode="valid"
        )

        return self.medias

    def prever(self, quantidade_dias=1):

        if self.precos is None:
            raise ValueError("O modelo ainda não foi ajustado.")

        if quantidade_dias < 1:
            raise ValueError("A quantidade de dias deve ser positiva.")

        previsoes = []

        # Utilizar os últimos preços disponíveis
        historico = list(self.precos)

        for _ in range(quantidade_dias):

            ultimos_precos = historico[-self.periodo:]

            proximo_preco = np.mean(ultimos_precos)

            previsoes.append(proximo_preco)

            # Adicionar previsão ao histórico
            historico.append(proximo_preco)

        return np.array(previsoes)

    def obter_ultima_media(self):

        if self.medias is None:
            raise ValueError("O modelo ainda não foi ajustado.")

        return self.medias[-1]

    def plotar(self, dias_previsao=0):

        if self.precos is None:
            raise ValueError("O modelo ainda não foi ajustado.")

        tempo = np.arange(len(self.precos))

        tempo_medias = np.arange(
            self.periodo - 1,
            len(self.precos)
        )

        previsoes = self.prever(dias_previsao) if dias_previsao > 0 else []

        plt.figure(figsize=(12, 6))

        plt.plot(
            tempo,
            self.precos,
            label="Preços reais"
        )

        plt.plot(
            tempo_medias,
            self.medias,
            label=f"Média móvel ({self.periodo} períodos)"
        )

        if dias_previsao > 0:

            tempo_futuro = np.arange(
                len(self.precos),
                len(self.precos) + dias_previsao
            )

            plt.plot(
                tempo_futuro,
                previsoes,
                linestyle="--",
                label="Previsão"
            )

        plt.title("Média Móvel - Preço da Ação")
        plt.xlabel("Dias de negociação")
        plt.ylabel("Preço (R$)")

        plt.legend()
        plt.grid()
        plt.show()
