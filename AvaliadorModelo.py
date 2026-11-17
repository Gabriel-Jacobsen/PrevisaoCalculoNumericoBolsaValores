
import numpy as np


class AvaliadorModelos:

    @staticmethod
    def calcular_metricas(valores_reais, previsoes):

        reais = np.asarray(valores_reais, dtype=float)
        previstos = np.asarray(previsoes, dtype=float)

        if len(reais) != len(previstos):
            raise ValueError("Os vetores precisam ter o mesmo tamanho.")

        if not np.all(np.isfinite(reais)) or not np.all(np.isfinite(previstos)):
            raise ValueError("Existem valores inválidos.")

        erros = reais - previstos

        mae = np.mean(np.abs(erros))

        rmse = np.sqrt(np.mean(erros ** 2))

        ss_res = np.sum(erros ** 2)
        ss_tot = np.sum((reais - np.mean(reais)) ** 2)

        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

    @staticmethod
    def comparar(resultados):

        for nome, metricas in resultados.items():

            print(f"\n{nome}")
            print(f"MAE:  {metricas['MAE']:.4f}")
            print(f"RMSE: {metricas['RMSE']:.4f}")
            print(f"R²:   {metricas['R2']:.4f}")
