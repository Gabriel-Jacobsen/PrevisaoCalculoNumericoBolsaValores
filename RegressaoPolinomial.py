
import numpy as np
import matplotlib.pyplot as plt


class RegressaoPolinomial:

    def __init__(self, grau=2):

        self.grau = grau
        self.coeficientes = None
        self.r2 = None

    def ajustar(self, precos):

        precos = np.array(precos, dtype=float)

        if len(precos) <= self.grau:
            raise ValueError(
                "Quantidade de dados insuficiente para o grau escolhido."
            )

        if not np.all(np.isfinite(precos)):
            raise ValueError("Os preços contêm valores inválidos.")

        # Criar variável de tempo
        tempo = np.arange(len(precos))

        # Calcular coeficientes pelo método dos mínimos quadrados
        self.coeficientes = np.polyfit(
            tempo,
            precos,
            self.grau
        )

        # Calcular valores estimados
        previsoes = self.prever(tempo)

        # Calcular R²
        soma_residuos = np.sum((precos - previsoes) ** 2)
        soma_total = np.sum((precos - np.mean(precos)) ** 2)

        self.r2 = (
            1 - soma_residuos / soma_total
            if soma_total != 0 else 1.0
        )

        return self.coeficientes

    def prever(self, tempo):

        if self.coeficientes is None:
            raise ValueError("O modelo ainda não foi ajustado.")

        return np.polyval(self.coeficientes, tempo)

    def obter_equacao(self):

        if self.coeficientes is None:
            raise ValueError("O modelo ainda não foi ajustado.")

        termos = []

        grau_atual = self.grau

        for coeficiente in self.coeficientes:

            if grau_atual == 0:
                termo = f"{coeficiente:.4f}"

            elif grau_atual == 1:
                termo = f"{coeficiente:.4f}t"

            else:
                termo = f"{coeficiente:.4f}t^{grau_atual}"

            termos.append(termo)
            grau_atual -= 1

        return "P(t) = " + " + ".join(termos)

    def plotar(self, precos, dias_previsao=0):

        precos = np.array(precos, dtype=float)

        tempo = np.arange(len(precos))

        tempo_completo = np.arange(
            len(precos) + dias_previsao
        )

        previsoes = self.prever(tempo_completo)

        plt.figure(figsize=(12, 6))

        plt.scatter(
            tempo,
            precos,
            label="Preços reais"
        )

        plt.plot(
            tempo_completo,
            previsoes,
            label=f"Regressão polinomial (grau {self.grau})"
        )

        plt.title("Regressão Polinomial - Preço da Ação")
        plt.xlabel("Dias de negociação")
        plt.ylabel("Preço (R$)")

        plt.legend()
        plt.grid()
        plt.show()
