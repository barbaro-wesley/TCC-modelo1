# Auditoria do fluxo nacional de uma semana — TCC

Data de conclusão: 12/09/2026. Repositório auditado: `C:\Users\Wesley.Barbaro\Documents\tcc-joao\tcc-mvp`. Commit: `d807696953e2d4140b66b79aa76d12913e1e8840`.

## Parecer

O repositório é uma base adequada para continuar o TCC, com ARIMA como referência operacional, persistência como baseline e VS-ePL-KRLS como modelo experimental. O fluxo atual de horizonte semanal passou nas verificações de alinhamento temporal e de uso dos alvos. **Não encontrei vazamento direto de alvos futuros nas partes verificadas do treinamento h=1.** Isso não certifica todo o repositório nem a disponibilidade histórica dos dados na data de emissão.

As principais pendências estão na validade da avaliação: a decisão final de produção ainda depende do holdout; as métricas históricas não representam automaticamente o protocolo corrigido; e o intervalo hoje servido usa uma regra diferente daquela que gerou a cobertura publicada. Portanto, os resultados existentes devem ser apresentados como avaliação retrospectiva do protocolo anterior, sem afirmar que são uma validação independente da versão atual.

Durante a auditoria, o repositório recebeu alterações externas, passando do commit `da7731c` ao commit acima. Este parecer considera a versão mais recente. Não trata como pendentes os problemas que ela já corrigiu. A alteração local preexistente em `src/vs_epl_krls/api.py` foi preservada; a API e o ambiente publicado não foram auditados integralmente.

## Como o treinamento funciona

1. A entrada é a planilha local da ANP, filtrada para Diesel B S10 e agregada por produto e data. A série carregada contém **705 observações**, de **30/12/2012 a 30/08/2026**. O objetivo é prever o preço médio nacional de revenda da semana seguinte.
2. A preparação constrói uma grade semanal explícita. O alvo é localizado pela data `origem + 7 dias`; semanas ausentes permanecem ausentes na grade. Amostras sem os atributos ou alvos necessários são retiradas do conjunto supervisionado.
3. Os atributos endógenos disponíveis incluem preço, defasagens e dinâmica histórica. O candidato salvo usa defasagens de 0, 1, 2, 4, 8 e 12 semanas e aprende a variação do preço (`delta`). O preço corrente é legítimo somente se já estiver publicado no momento em que a previsão é emitida.
4. A validação usa três blocos temporais de 52 semanas. Cada bloco começa com ajuste usando o passado; a simulação prevê e incorpora os alvos à medida que suas datas se tornam conhecidas. A normalização é ajustada no trecho de treinamento. O holdout tem 104 semanas, fixadas por calendário, de **18/08/2024 a 09/08/2026**.
5. A seleção compara VS-ePL-KRLS, ARIMA, Ridge, persistência e ensemble. O script atual prioriza desenvolvimento e exige `--evaluate-holdout` para reabrir o holdout já utilizado. O padrão de normalização novo é `robust_bounded`; o manifesto legado ainda seleciona um candidato `minmax`.
6. O treinamento de produção lê o manifesto escolhido e os resíduos de calibração, ajusta um novo bundle com o histórico disponível, aquece o parâmetro adaptativo do intervalo e salva o artefato. A atualização semanal prevê antes de aprender a nova observação e exige a data exata do alvo esperado.

Referências: [construção do alvo](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/src/vs_epl_krls/selection.py:213), [validação temporal](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/src/vs_epl_krls/selection.py:447) e [treinamento de produção](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/06_train_s10_production.py:58).

## Verificações executadas

| Verificação | Resultado observado |
|---|---|
| Distância entre origem e alvo no supervisionado | Nenhuma amostra com distância diferente de sete dias |
| Alvos disponíveis nas origens avaliadas | Nenhum alvo incorporado com data posterior à origem |
| Alteração artificial dos dez preços finais em +R$ 100 | Atributos anteriores de `price`, `lags` e `dynamics` permaneceram idênticos |
| Alteração dos alvos do último bloco de desenvolvimento em +R$ 50 | Primeira previsão permaneceu igual; a seguinte mudou após a chegada do novo alvo |
| Atualização online saltando uma semana | Rejeitada; nenhuma amostra acrescentada ao monitor de cobertura |
| Carregamento do bundle com SHA-256 verificado | Previsão exatamente igual ao manifesto salvo |
| Integridade do ledger de produção | Cadeia de hashes válida, com três registros |
| Estabilidade dos arquivos examinados pelo verificador | Os oito arquivos monitorados mantiveram seus hashes durante a execução |
| Testes direcionados do repositório | **106 testes e 19 subtestes passaram**, em 32,37 segundos |

Também foi reexecutado o último bloco de desenvolvimento, de **20/08/2023 a 11/08/2024**, com os parâmetros do candidato legado e o código atual: 52 previsões, RMSE **0,063788** e MAE **0,029561 R$/L**. Esse resultado verifica execução e causalidade; não constitui nova seleção nem demonstra superioridade sobre outros modelos.

Não foi refeita a busca completa de hiperparâmetros. As previsões salvas do holdout foram lidas para conferência de alvos e cálculo aritmético da cobertura; não houve nova rodada de ajuste sobre esse holdout. Testes com perturbação foram delimitados aos atributos endógenos e ao candidato informado acima.

## Pendências encontradas

### 1. Alta — o holdout participa da escolha final

O candidato VS é selecionado em desenvolvimento, mas o modelo final de produção pode ser substituído por ARIMA conforme as métricas do holdout. O script de treinamento repete essa regra. Em uma prova controlada da função `_approved_primary`, mantendo o manifesto igual e alterando somente o RMSE do ensemble de 1,01 para 1,03, com ARIMA em 1,00, a escolha mudou de ensemble para ARIMA.

Isso é uso do conjunto de teste na decisão final, mesmo sem ajustar diretamente os coeficientes com seus alvos. A afirmação `selected_on_validation_only` não descreve toda a decisão. Um critério de promoção previamente definido pode ser útil operacionalmente, mas o mesmo conjunto deixa de ser uma avaliação intocada do modelo selecionado por esse critério.

**Encaminhamento:** congelar em desenvolvimento a escolha, os hiperparâmetros, os pesos e as regras de fallback. Identificar o holdout existente como já consultado e reservar dados futuros, com previsões registradas previamente, para confirmação independente.

Evidências: [gates da seleção](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/05_s10_model_selection.py:397) e [aprovação na produção](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/06_train_s10_production.py:46).

### 2. Alta — métricas antigas acompanham um protocolo modificado

O código atual corrige o calendário, a disponibilidade dos alvos, o reaproveitamento de resultados por fingerprint e a equivalência do ARIMA de avaliação com o ajuste de produção. Entretanto, os argumentos padrão do treinamento ainda apontam para `s10_selection/selection_manifest_h1.json` e seus resíduos históricos. Esse manifesto não contém `pipeline_version` nem `validation_fingerprint` e representa o candidato com MinMax, enquanto a seleção nova usa outro padrão.

A model card atual combina treinamento até 30/08/2026 com a tabela histórica de 104 semanas. As métricas dessa tabela continuam sendo resultados do experimento salvo, mas não demonstram, por si só, o desempenho do treinamento corrigido. Reajustar o modelo com todos os dados conhecidos para prever uma semana posterior é legítimo; o problema é atribuir a esse novo protocolo a avaliação de outro.

**Encaminhamento:** vincular cada tabela ao código, à configuração, aos dados, às previsões e aos resíduos que a produziram. Conferir compatibilidade antes de treinar e avaliar o fluxo corrigido em desenvolvimento com a mesma rotina usada em produção. O Ridge ainda tem periodicidade de reajuste diferente entre a avaliação e as atualizações de produção; se permanecer na comparação ou no ensemble, deve ser harmonizado.

Evidências: [caminhos padrão](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/06_train_s10_production.py:220), [manifesto legado](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_selection/selection_manifest_h1.json) e [model card atual](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_production/model_card.md).

### 3. Alta — cobertura publicada e intervalo atual usam regras diferentes

A cobertura histórica foi recalculada: **96 de 104 observações, ou 92,31%**, com largura **0,120969 R$/L**, usando quantis fixos 10% e 90% de 156 resíduos. O cálculo confere para essa regra.

O bundle atual usa alpha adaptativo aquecido de **0,344**, portanto consulta quantis de **17,2% e 82,8%**, além de aplicar limites para incluir o ponto previsto e tratar fallback. Para o mesmo ponto de R$ 6,88/L:

| Regra | Limite inferior | Limite superior |
|---|---:|---:|
| Quantis fixos 10%–90% dos resíduos do bundle | 6,816471 | 6,937440 |
| Intervalo atualmente produzido pelo bundle | 6,840635 | 6,902204 |

Aquecimento repetido na cópia em memória manteve o intervalo, pois o artefato já estava aquecido. O contador de cobertura online desse bundle está em zero. Não há evidência aqui de que o intervalo adaptativo atual alcance a cobertura histórica de 92,31%. Alpha também não deve ser convertido diretamente em uma cobertura empírica garantida. Os campos `p10` e `p90` ficaram semanticamente imprecisos depois da adaptação dos quantis.

**Encaminhamento:** avaliar causalmente a própria rotina de intervalo de produção, incluindo aquecimento, atualizações e fallback, e publicar sua cobertura, largura e número de observações separadamente da regra antiga. Vincular os resíduos à versão do experimento e às respectivas datas.

Evidências: [intervalo na avaliação](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/05_s10_model_selection.py:442), [aquecimento e previsão](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/src/vs_epl_krls/production.py:307) e [reutilização dos diagnósticos na model card](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/06_train_s10_production.py:139).

### 4. Alta para alegações prospectivas — falta comprovar disponibilidade e emissão

O treinamento trabalha com datas das semanas. Isso não comprova a data em que o preço agregado ficou disponível, nem preserva necessariamente a versão histórica que teria sido consultada em cada origem. Assim, a causalidade verificada é relativa às datas da série, condicionada à disponibilidade do preço corrente.

O ledger nacional tem um `bootstrap_forecast` registrado em **24/08/2026 às 02:23 UTC**, para a semana iniciada em **16/08/2026**, seguido da observação oficial aproximadamente um segundo depois. Essa semana já havia terminado. A cadeia de hashes está íntegra, mas não comprova que a previsão foi registrada antes da realização. O próprio script permite inicializar o registro no processo de ingestão que já dispõe da observação oficial. Isso não demonstra fabricação do resultado; limita o uso desse registro como prova prospectiva.

As três semanas posteriores ao corte do holdout, chamadas `prospective_weeks` no manifesto de janelas, são somente uma contagem por datas. Elas não equivalem automaticamente a três previsões prospectivas válidas.

**Encaminhamento:** definir horário de emissão, disponibilidade das fontes e semana-alvo; armazenar `issued_at`, `published_at` quando disponível, snapshot da fonte, hash do artefato e previsão antes de conhecer o resultado. Classificar bootstrap e replay separadamente. Uma semana entre rótulos não garante sete dias completos de antecedência operacional.

Evidência: [ingestão e bootstrap](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/14_s10_ingest_official.py:32).

### 5. Média — modelo treinado e ponteiro de release são diferentes

O bundle local e seu manifesto são consistentes: versão **1.2.0**, SHA-256 `bb7d0aa61f3df2f633bd00983538600b66b9f32edac019659264698ebc036550`, última observação em 30/08/2026 e previsão para 06/09/2026 de **R$ 6,88/L**. O primário é ARIMA; essa previsão específica coincide com a persistência.

Já `latest_release.json` aponta para outro artefato, com alvo em **23/08/2026** e última observação em 16/08/2026. Isso pode refletir uma promoção separada e intencional, mas treinar o arquivo base não atualiza automaticamente a versão referenciada pela release. O runner semanal declara que não promove nem retreina a release nacional servida. Não foi verificado qual artefato está efetivamente carregado em um serviço externo.

**Encaminhamento:** tornar explícito qual versão será demonstrada no TCC e conferir a promoção pelo procedimento existente antes de demonstrar a aplicação. Os avisos do challenger — pressão de dicionário, substituições e beta próximo do piso — justificam mantê-lo em avaliação; eles não demonstram vazamento.

Evidências: [ponteiro local](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_product/latest_release.json) e [escopo do runner semanal](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/scripts/weekly_refresh.py:11).

## Próxima etapa proposta

Priorizar uma avaliação em desenvolvimento que execute o mesmo bundle da produção, com escolha prévia dos modelos, calendário, reajustes, fallback e intervalos. Publicar resultados identificados como protocolo corrigido e preservar a tabela anterior como histórico. Em seguida, congelar o protocolo e começar o registro prospectivo válido, sem selecionar novas configurações pelos resultados do holdout já consultado.

Para a redação atual: apresentar ARIMA como primário do artefato auditado, persistência como comparação obrigatória e VS-ePL-KRLS como experimento. Não afirmar superioridade da versão corrigida nem ausência universal de vazamento antes dessas etapas.

## Reprodução e arquivos entregues

O código do repositório auditado e seus artefatos não foram alterados por esta auditoria. Os arquivos abaixo foram criados no workspace `MachineLearning-MVP`.

- [Evidências completas, hashes e resultados](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/reports/auditoria_nacional_h1/evidencias.json)
- [Verificador reproduzível](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/reports/auditoria_nacional_h1/verificar.py)
- [Registro da execução dos testes](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/reports/auditoria_nacional_h1/testes.txt)

Ambiente do verificador: Python 3.14.5, NumPy 2.5.0, pandas 3.0.5. Executar o comando abaixo no workspace que contém este relatório, usando o interpretador com as dependências instaladas. O parâmetro `--commit` é uma identificação fornecida externamente: conferir `git rev-parse HEAD` no repositório auditado antes de repetir, pois o script não valida esse parâmetro sozinho.

```powershell
python reports/auditoria_nacional_h1/verificar.py C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp --commit d807696953e2d4140b66b79aa76d12913e1e8840
```

O script grava evidências ao lado de si, usa cópias em memória para as perturbações e verifica hashes antes e depois. Ele não executa downloads, promoção, seleção completa ou atualização dos artefatos originais. Os resultados do verificador devem ser lidos junto com a inspeção de código e os testes; não constituem uma prova formal da ausência de todo possível vazamento.
