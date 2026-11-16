
import numpy as np


class RegressaoLinear:

    def __init__(self):
        self.coeficiente_angular = None
        self.coeficiente_linear = None
        self.r2 = None

    def ajustar(self, precos):

        # Converter os preços para um array NumPy
        precos = np.array(precos, dtype=float)

        if len(precos) < 2:
            raise ValueError("São necessários pelo menos dois valores.")

        if not np.all(np.isfinite(precos)):
            raise ValueError("Os preços contêm valores inválidos.")

        # Criar variável de tempo
        tempo = np.arange(len(precos))

        # Calcular médias
        media_tempo = np.mean(tempo)
        media_precos = np.mean(precos)

        # Calcular coeficientes pelo método dos mínimos quadrados
        numerador = np.sum(
            (tempo - media_tempo) * (precos - media_precos)
        )

        denominador = np.sum(
            (tempo - media_tempo) ** 2
        )

        self.coeficiente_angular = numerador / denominador

        self.coeficiente_linear = (
            media_precos -
            self.coeficiente_angular * media_tempo
        )

        # Calcular R²
        previsoes = self.prever(tempo)

        soma_residuos = np.sum((precos - previsoes) ** 2)
        soma_total = np.sum((precos - media_precos) ** 2)

        self.r2 = (
            1 - soma_residuos / soma_total
            if soma_total != 0 else 1.0
        )

        return self.coeficiente_linear, self.coeficiente_angular

    def prever(self, tempo):

        if self.coeficiente_angular is None:
            raise ValueError("O modelo ainda não foi ajustado.")

        tempo = np.array(tempo)

        return (
            self.coeficiente_linear +
            self.coeficiente_angular * tempo
        )

    def obter_equacao(self):

        return (
            f"P(t) = {self.coeficiente_linear:.4f} "
            f"+ {self.coeficiente_angular:.4f}t"
        )
