
import yfinance as yf
import pandas as pd


class DadosBolsa:

    def __init__(self, codigo_acao, data_inicio, data_fim, intervalo="1d"):

        self.codigo_acao = codigo_acao
        self.data_inicio = data_inicio
        self.data_fim = data_fim
        self.intervalo = intervalo

        self.dados = None

    def buscar_dados(self):

        acao = yf.Ticker(self.codigo_acao)

        self.dados = acao.history(
            start=self.data_inicio,
            end=self.data_fim,
            interval=self.intervalo,
            auto_adjust=True
        )

        if self.dados.empty:
            raise ValueError("Nenhum dado encontrado para o período informado.")

        return self.dados

    def obter_fechamentos(self):

        if self.dados is None:
            self.buscar_dados()

        return self.dados["Close"]

    def salvar_csv(self, nome_arquivo="dados_bolsa.csv"):

        if self.dados is None:
            self.buscar_dados()

        self.dados.to_csv(nome_arquivo)

        print(f"Dados salvos em {nome_arquivo}")
