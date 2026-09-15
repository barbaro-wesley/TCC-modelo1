# Correções temporais do pipeline principal

Escopo: `src/`, `scripts/` e `tests/`. A cópia histórica em
`reports/auditoria_nacional_h1/implementacao/` permanece intacta e não é utilizada
pela API. Este documento não substitui os resultados da auditoria original.

## Protocolo

Identificador: `calendar-mature-labels-v1`.

Cada par supervisionado liga a data da origem à data do alvo. Um rótulo ou resíduo
só pode ser usado quando sua data de alvo é menor ou igual à origem da previsão.
Essa igualdade pressupõe emissão **após a publicação** da observação da origem;
`data` continua identificando a semana/mês da observação, não um instante real de
publicação. O backtest não deve ser descrito como previsão emitida no começo dessa
semana. A comprovação de disponibilidade histórica exige dados point-in-time.

## Mudanças

| Achado | Correção |
| --- | --- |
| Alvos mensais h=6/h=12 antecipados | Treino inicial contém somente rótulos maduros. Atualizações são liberadas pela data do alvo, inclusive quando pares com NaN foram excluídos. Sem atualização online, mantém-se apenas o ajuste inicial maduro. |
| Resíduos futuros nos intervalos h=2/h=4 | Quantis usam apenas resíduos cujo alvo já foi observado; a calibração recebe datas e horizonte. |
| Cobertura desalinhada | Mesma máscara finita para alvo e dois limites; comprimentos divergentes geram erro, sem truncamento silencioso. |
| Seleção de features consultando toda a amostra | Lista fixa de 18 features. ULSD permanece excluído por indisponibilidade da fonte; colunas obrigatórias ausentes causam erro. |
| Lacunas tratadas como semanas consecutivas | Lags/janelas reiniciam a cada lacuna. Só se aceitam pares com todas as semanas intermediárias presentes. O painel preserva `target_date`. |
| Cortes de treino em painéis filtrados | VS-ePL-KRLS, GBM, LSTM e helpers genéricos usam a data do alvo, não a distância entre linhas após filtros. |
| Históricos ARIMA/ARIMAX e média móvel comprimidos | Usam somente o bloco semanal contínuo de cada origem. ARIMA/ARIMAX reiniciam após quebra; o preço não é imputado. |
| Inferência LSTM deslocada em h>1 | O corte de rótulos de treino é separado da janela de inferência `X[t-seq_len:t]`, correspondente a `y[t]`. Sequências de treino/inferência não atravessam lacunas. |
| ARIMAX de produção uma semana atrasado | Ajusta pelo histórico observado completo, incluindo a última semana, e usa o horizonte solicitado. |
| Produção silenciosamente recuando a uma linha antiga | Última linha sem features completas gera erro; removidos preenchimentos retroativos. |
| Importância em dados usados pelo modelo final | Modelo diagnóstico separado, congelado na primeira origem de avaliação, mede importância em entradas posteriores fora do seu treino. Não representa o modelo final reajustado. |
| Comparação DM com amostras potencialmente diferentes | Erros do candidato e do naive são pareados pela mesma máscara de datas. |
| Mistura com métricas antigas | Ranking e resíduos exigem o novo identificador de protocolo. Publicação com ranking h=1 rejeita outros horizontes. |

As políticas de lacunas e de features são conservadoras e alteram as amostras
avaliadas. Espera-se que contagens, métricas, modelos selecionados e previsões
mudem após a reexecução. Não se devem comparar números antigos e novos como se
tivessem sido obtidos no mesmo conjunto de origens.

## Verificação

```bash
python -m pip install pytest
PYTHONPATH=src python -m pytest tests -q
```

O `pytest.ini` da raiz limita a descoberta à suíte científica em `tests/`, evitando
coletar as cópias históricas. A suíte do backend continua separada:
`python -m pytest -c api/pytest.ini api/tests -q`.

Os testes usam séries sintéticas e modelos instrumentados para verificar quais
rótulos cada ajuste recebeu, invariância a perturbações futuras, datas de alvos,
lacunas, igualdade das máscaras, sequência LSTM (inclusive o entrypoint do
subprocesso), última observação ARIMAX e rejeição de métricas antigas. Um smoke
test percorre os três horizontes, exporta os artefatos em diretório temporário e
monta o payload de produção com estimadores instrumentados. Os testes existentes
do VS-ePL-KRLS também são preservados. Isso não equivale a retreinar todos os
estimadores reais nem a validar a acurácia preditiva.

Ambiente local de verificação: Python 3.12, NumPy 2.3.5, pandas 2.2.3,
SciPy 1.17.0, scikit-learn 1.8.0, statsmodels 0.14.4 e pytest 8.3.5.
Não corresponde integralmente ao `requirements.txt`; a matriz completa de
dependências fixadas e o treinamento real de PyTorch/GBM ainda precisam ser
verificados no ambiente científico do projeto.

## Reexecução necessária, não realizada por esta alteração

Em um ambiente de experimentação com as dependências científicas instaladas e as
fontes disponíveis, reconstruir os dados e executar:

```bash
PYTHONPATH=src python scripts/01_download.py
PYTHONPATH=src python scripts/02_reproducao.py
PYTHONPATH=src python scripts/03_semanal.py
```

Esses comandos baixam/reconstroem dados e substituem resultados locais. Preservar
os resultados antigos antes de executá-los. Para uma atualização operacional,
o job `05_atualizacao_semanal.py --forcar` refaz o backtest semanal e publica os
arquivos locais da API/README; isso é uma etapa posterior, não executada aqui.

## Limitações ainda abertas

- Datas e versões reais de publicação/revisão de ANP e exógenas não estão
  disponíveis no histórico. Não foi criada nem presumida evidência point-in-time.
- A escolha do vencedor continua usando o mesmo backtest do ranking divulgado.
  O menor erro é exploratório; é necessário teste temporal independente ou
  simulação causal da seleção para avaliar o processo de escolha.
- A calibração por quantis de resíduos não garante cobertura nominal de 80% em
  séries dependentes. Cobertura deve ser medida novamente.
- Fidelidade matemática ao artigo, competitividade do VS-ePL-KRLS, parametrização
  dos estimadores, operação do cron e deploy não são validados por esta correção.
- Os relatórios anteriores e a previsão no README são históricos e permanecem
  sem reavaliação integral. Não há nova alegação de reprodução científica.
