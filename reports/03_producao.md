# Bloco 3 — Modelo final de producao

**Recomendado (h=1 semana, criterio RMSE walk-forward): ARIMA**

Modelo recomendado para producao com base no RMSE walk-forward de 1 semana. Se VS-ePL-KRLS nao for o vencedor, ele permanece como candidato evolutivo porque atualiza a cada observacao e fornece sinais de drift via beta e regras.

## Ranking h=1

| model | rmse | mae | smape | dir_acc | coverage_p10_p90 |
| --- | --- | --- | --- | --- | --- |
| ARIMA | 0.07256 | 0.02785 | 0.56624 | 0.63704 | 0.00169 |
| ARIMAX | 0.07487 | 0.02936 | 0.59230 | 0.63519 | 0.00339 |
| naive | 0.07855 | 0.03068 | 0.62505 | 0.00000 | 0.00339 |
| media_movel | 0.13877 | 0.06737 | 1.36390 | 0.29444 | 0.01695 |
| XGBoost | 0.23956 | 0.13519 | 2.74902 | 0.40370 | 0.06780 |
| LightGBM | 0.24782 | 0.14226 | 2.97159 | 0.39630 | 0.07627 |
| VS-ePL-KRLS | 3.41189 | 3.23983 | 129.90292 | 0.50556 | 0.17966 |
| LSTM | 4.09207 | 3.83289 | 138.93798 | 0.50556 | 0.18136 |

## Previsao da proxima semana (preco medio nacional de revenda, R$/L)

- Semana observada: 2026-08-16
- Preco observado: 6.89
- Previsao pontual: 6.8820506644385055
- P10: 6.848318267811069
- P90: 6.899026522449789
- Prob. alta / estavel / queda: {'p_alta': 0.075, 'p_estavel': 0.6000000000000001, 'p_queda': 0.325}

## Model card

- Alvo: preco medio nacional de *revenda* do Diesel B S-10 (ANP), nao o preco de um posto.
- Horizonte de producao: 1 semana a frente.
- Frequencia de atualizacao: incremental a cada nova semana da ANP.
- Exogenas: Brent, USD/BRL, diesel internacional (se disponivel), defasagens, medias moveis, volatilidade, proxy de reajuste Petrobras.
- Preco de distribuicao: usado so na reproducao do artigo; serie ANP termina em ago/2020.
- Intervalo P10-P90: quantis conformais dos residuos walk-forward, nao intervalos gaussianos.
- Limitacoes: buraco ANP ago-out/2020; mudancas de politica de precos; o modelo nao antecipa reajuste da Petrobras no mesmo dia em que e anunciado se a semana ainda nao fechou.
- Quando reajustar: alerta Page-Hinkley, PSI alto nas exogenas, ou degradacao do RMSE movel de 12 semanas.

Este bloco nao declara reproducao do artigo. A reproducao esta no relatorio 01.