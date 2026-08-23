# Bloco 2 — Adaptacao semanal

Validacao walk-forward temporal (sem divisao aleatoria).
VS-ePL-KRLS atualiza de forma incremental. Modelos em lote reajustam a cada 4 semanas (LSTM a cada 8).
Features apenas defasadas. Preco de distribuicao NAO entra no modelo de producao apos ago/2020.

## Resultados

| horizon | model | rmse | mae | smape | dir_acc | coverage_p10_p90 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | VS-ePL-KRLS | 3.41189 | 3.23983 | 129.903 | 0.50556 | 0.17966 |
| 1 | naive | 0.07855 | 0.03068 | 0.62505 | 0.00000 | 0.00339 |
| 1 | media_movel | 0.13877 | 0.06737 | 1.36390 | 0.29444 | 0.01695 |
| 1 | ARIMA | 0.07256 | 0.02785 | 0.56624 | 0.63704 | 0.00169 |
| 1 | ARIMAX | 0.07487 | 0.02936 | 0.59230 | 0.63519 | 0.00339 |
| 1 | LightGBM | 0.24782 | 0.14226 | 2.97159 | 0.39630 | 0.07627 |
| 1 | XGBoost | 0.23956 | 0.13519 | 2.74902 | 0.40370 | 0.06780 |
| 1 | LSTM | 4.09207 | 3.83289 | 138.938 | 0.50556 | 0.18136 |
| 2 | VS-ePL-KRLS | 3.38590 | 3.20557 | 128.326 | 0.47304 | 0.19015 |
| 2 | naive | 0.13149 | 0.05759 | 1.16518 | 0.00000 | 0.01019 |
| 2 | media_movel | 0.17993 | 0.09190 | 1.86266 | 0.29913 | 0.03565 |
| 2 | ARIMA | 0.12478 | 0.05374 | 1.08281 | 0.61739 | 0.00679 |
| 2 | ARIMAX | 0.12800 | 0.05566 | 1.11609 | 0.63304 | 0.00679 |
| 2 | LightGBM | 0.28795 | 0.17506 | 3.62538 | 0.38087 | 0.07810 |
| 2 | XGBoost | 0.28804 | 0.17236 | 3.54151 | 0.40870 | 0.11205 |
| 2 | LSTM | 4.09480 | 3.83618 | 138.988 | 0.47304 | 0.18166 |
| 4 | VS-ePL-KRLS | 3.40252 | 3.22757 | 129.413 | 0.45128 | 0.19250 |
| 4 | naive | 0.21054 | 0.10682 | 2.16267 | 0.00000 | 0.04429 |
| 4 | media_movel | 0.24693 | 0.13793 | 2.79862 | 0.29402 | 0.07325 |
| 4 | ARIMA | 0.20515 | 0.10252 | 2.06533 | 0.61880 | 0.03918 |
| 4 | ARIMAX | 0.21013 | 0.10607 | 2.12658 | 0.64103 | 0.03237 |
| 4 | LightGBM | 0.35911 | 0.23099 | 4.82634 | 0.41368 | 0.10221 |
| 4 | XGBoost | 0.34929 | 0.22480 | 4.63261 | 0.40000 | 0.09710 |
| 4 | LSTM | 4.10032 | 3.84281 | 139.088 | 0.45128 | 0.17888 |

## Melhor por horizonte (RMSE)

| horizon | model | rmse | mae | smape | dir_acc |
| --- | --- | --- | --- | --- | --- |
| 1 | ARIMA | 0.07256 | 0.02785 | 0.56624 | 0.63704 |
| 2 | ARIMA | 0.12478 | 0.05374 | 1.08281 | 0.61739 |
| 4 | ARIMA | 0.20515 | 0.10252 | 2.06533 | 0.61880 |

## Previsao 1 semana a frente (ultimo ponto da amostra)

```json
{
  "ultima_semana_observada": "2026-08-16",
  "preco_observado_ultima_semana": 6.89,
  "horizonte": "1 semana",
  "previsao_pontual": 4.961981822781968,
  "p10": 6.820331366425106,
  "p90": 9.967862530902257,
  "probabilidades": {
    "p_alta": 0.775,
    "p_estavel": 0.03749999999999998,
    "p_queda": 0.1875
  },
  "n_regras": 1,
  "aviso": "Previsao do preco medio nacional de REVENDA. Nao e preco de bomba de um posto especifico."
}
```

Os numeros do bloco 1 (artigo mensal 2012-2020) nao se transferem para este bloco.