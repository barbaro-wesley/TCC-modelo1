# Protocolo congelado do TCC — nacional, uma semana

Versão: `national-h1-production-replay-v1`. Preparado em 12/09/2026.

Este é o fluxo de desenvolvimento para a comparação corrigida do TCC. Usa a implementação do bundle de produção e mantém o holdout já consultado fora do avaliador. O congelamento é feito agora: não constitui pré-registro anterior à consulta histórica do holdout.

## Experimento fixado

| Item | Decisão |
|---|---|
| Série | Preço médio nacional de revenda do Diesel B S10, ANP, em R$/L |
| Horizonte | Alvo por data: origem + sete dias |
| Modelos comparados | ARIMA, persistência e VS-ePL-KRLS |
| Entradas do VS | Defasagens 0, 1, 2, 4, 8 e 12; alvo em variação; normalização `robust_bounded` |
| Hiperparâmetros | Explícitos no JSON; herdados do candidato histórico de desenvolvimento com a normalização corrigida; nenhuma nova busca nesta etapa |
| Calibração inicial | 26 semanas consecutivas anteriores a cada bloco, mesmas datas para os três modelos |
| Janela móvel de resíduos | Até 156 observações, conforme o bundle |
| Mudanças anômalas observadas | Rejeitadas pela proteção de produção, sem aprovação automática ou descarte silencioso |
| Escolha e promoção | Este avaliador não escolhe nem promove modelos |
| Evidência nova | Desenvolvimento; nunca rotulada como teste final independente |

Os três blocos de 52 semanas-alvo são:

| Bloco | Início | Fim |
|---|---|---|
| validation_1 | 22/08/2021 | 14/08/2022 |
| validation_2 | 21/08/2022 | 13/08/2023 |
| validation_3 | 20/08/2023 | 11/08/2024 |

O snapshot de desenvolvimento termina em 11/08/2024 e contém 598 preços observados. As semanas do holdout histórico, 18/08/2024 a 09/08/2026, e a cauda posterior não são incluídas nesse snapshot. O hash da planilha original identifica a fonte completa; a execução usa somente o JSON de desenvolvimento congelado.

A calibração foi fixada em 26 semanas porque 52 semanas anteriores ao primeiro bloco atravessam a lacuna de 2020. A pré-validação exige também 13 semanas anteriores consecutivas para os atributos. Essa escolha preserva os três blocos e não foi determinada por comparação de erros. Com 26 resíduos, a rotina de aquecimento do bundle mantém alpha nominal em 0,2, pois ela exige 40 resíduos para aquecer. As atualizações posteriores adaptam alpha normalmente. Portanto, a inicialização dos resíduos deste experimento difere daquela da release antiga, que usou resíduos de outro protocolo; o código de previsão e adaptação é compartilhado.

## Como avaliação e produção compartilham o fluxo

`S10ProductionForecaster.fit_calibrated` centraliza a construção, o ajuste de todos os componentes e o aquecimento do intervalo. O script de treinamento `06_train_s10_production.py` e o novo avaliador chamam essa mesma entrada.

Para cada bloco e cada modelo:

1. Ajustar um bundle apenas no histórico anterior à calibração.
2. Percorrer as 26 semanas anteriores, chamando `predict_next` antes de entregar a observação a `update_one`. Extrair resíduos fora da amostra do componente primário e da persistência, em fluxos separados, como espera o bundle.
3. Ajustar um bundle novo com os preços conhecidos até a origem do bloco e os resíduos anteriores, pelo inicializador compartilhado.
4. Para cada semana avaliada, registrar ponto, intervalo, alpha, quantidade de resíduos, fallback e fingerprint do histórico. Só então incorporar a observação por `update_one`.
5. Calcular métricas sobre as previsões efetivamente servidas, incluindo os efeitos do fallback. Os limites são chamados `lower` e `upper` nos resultados porque alpha adaptativo pode deixar de corresponder aos quantis 10% e 90%.

ARIMA e Ridge são reajustados com a periodicidade da produção, e o VS aprende incrementalmente. A normalização do VS permanece a ajustada na inicialização daquele bundle. Não há uma implementação separada de previsão, atualização, fallback ou intervalo dentro do avaliador. Ridge e ensemble continuam sendo componentes internos do bundle, mas não são modelos selecionáveis neste protocolo do TCC.

O procedimento usa uma cópia em memória e não carrega nem modifica os artefatos publicados. Uma falha, semana ausente ou anomalia interrompe o resultado e gera `status: failed`; não é convertida silenciosamente em previsão válida. Na etapa de congelamento, calendários sem suporte suficiente são recusados antes de criar o diretório.

## Congelamento e reprodução

Arquivos principais:

- [Configuração](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/configs/s10_nacional_h1.json)
- [Manifesto congelado](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/tcc_h1_protocol_v1/manifest.json)
- [Histórico de desenvolvimento](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/tcc_h1_protocol_v1/development_history.json)
- [Entrada do avaliador](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/34_s10_frozen_experiment.py)
- [Implementação](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/src/vs_epl_krls/experiment.py)

O manifesto registra configuração, versões de Python e dependências, hash da fonte bruta, hash do snapshot, hashes do código e identificação do commit base. Os hashes descrevem os arquivos efetivos, incluindo mudanças ainda não commitadas; o commit base sozinho não representa essas mudanças. O identificador do experimento é calculado sobre o conteúdo do manifesto. Ele é um mecanismo de detecção de alterações, não uma assinatura ou prova externa de emissão.

A avaliação recusa snapshot, manifesto, código ou ambiente diferentes dos congelados. A configuração é lida do manifesto na execução, não de um JSON editável ao lado. Atualizações da planilha original não alteram esse experimento. Mudanças deliberadas exigem outro diretório e outra versão; comandos nunca sobrescrevem um experimento ou resultado existente.

Executar no diretório do repositório, usando Python com as dependências instaladas. O congelamento inicial já foi produzido; para um experimento futuro, conferir o commit com `git rev-parse HEAD` e escolher um novo nome de saída:

```powershell
python scripts/34_s10_frozen_experiment.py freeze --source-commit COMMIT_CONFERIDO --output reports/NOVO_EXPERIMENTO
```

Verificação curta, com duas semanas do primeiro bloco:

```powershell
python scripts/34_s10_frozen_experiment.py evaluate --frozen reports/tcc_h1_protocol_v1 --output reports/NOVO_SMOKE --fold validation_1 --smoke-weeks 2
```

Próxima etapa: executar os três blocos completos, sem `--smoke-weeks`, em uma saída nova:

```powershell
python scripts/34_s10_frozen_experiment.py evaluate --frozen reports/tcc_h1_protocol_v1 --output reports/tcc_h1_development_v1
```

Esse comando fará os reajustes semanais reais dos três modelos e pode levar vários minutos. Produzirá `predictions.json`, `result.json` e `status.json`. O teste curto é rotulado `smoke_only`; seus erros não sustentam ranking ou alegações de desempenho. O comando completo de comparação não foi executado como parte dos itens 1 e 2.

## Limites preservados explicitamente

Esta implementação resolve o congelamento e fornece uma avaliação com o fluxo de produção compartilhado. Os scripts históricos de seleção e os gates de promoção continuam legados: não executar `05`/`06` esperando que consumam automaticamente este novo manifesto. O `06` passa a compartilhar a inicialização do bundle, mas sua escolha a partir do manifesto antigo não foi migrada nesta etapa. Uma futura promoção precisa consumir uma decisão tomada em desenvolvimento, sem reaproveitar o holdout para escolher novamente.

As datas das semanas não comprovam a disponibilidade real de cada divulgação. O experimento é retrospectivo e assume que o preço da origem já era conhecido. A comprovação prospectiva exigirá registrar horário de emissão, versão da fonte e previsão antes da publicação do resultado. Também permanece necessária a avaliação completa dos intervalos deste protocolo; a cobertura histórica de 92,3% não deve ser atribuída a ele.
