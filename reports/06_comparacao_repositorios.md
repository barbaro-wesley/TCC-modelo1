**Comparação de desempenho — MachineLearning-MVP e tcc-mvp**

Data da análise: 11/09/2026.

O VS-ePL-KRLS do segundo repositório, tcc-mvp, teve desempenho muito superior ao do primeiro: seu RMSE foi 96,91% menor na mesma janela de avaliação. Entretanto, os modelos escolhidos para produção são ARIMA nos dois projetos e ficaram praticamente empatados. O segundo projeto apresenta um processo de seleção e acompanhamento mais desenvolvido, mas isso não se traduziu em ganho relevante do ARIMA servido.

**Como a comparação foi feita**

O primeiro repositório é [MachineLearning-MVP](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/README.md). O segundo é [tcc-mvp](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/README.md).

Não comparei diretamente o RMSE geral de 0,072558 do primeiro com o RMSE de 0,081453 do segundo: eles foram calculados em períodos diferentes. O primeiro agrega 610 previsões; o segundo apresenta uma janela final de 104 semanas. Usar esses números sem alinhar as datas produziria uma conclusão enganosa.

Para comparar os mesmos casos, utilizei os arquivos de previsões individuais já existentes. No primeiro projeto, o campo data identifica a origem, e não a semana prevista. Recuperei a data do alvo pela próxima observação da série original e fiz a junção com target_date do segundo projeto. Nenhum modelo foi retreinado nesta comparação e nenhum arquivo do segundo repositório foi alterado.

As verificações confirmaram:

- 104 semanas-alvo consecutivas, de **18/08/2024 a 09/08/2026**;
- horizonte de exatamente sete dias em todos os pares dessa janela;
- o mesmo alvo: preço médio nacional de revenda do Diesel B S-10;
- valores reais idênticos, com diferença máxima igual a zero;
- previsões de persistência idênticas, com diferença máxima igual a zero;
- previsões numéricas finitas para todos os candidatos comparados.

Assim, as métricas abaixo foram recalculadas sobre os mesmos preços e semanas. Os protocolos de treinamento continuam diferentes: a comparação mede os sistemas que cada projeto efetivamente avaliou, e não um experimento em que apenas a implementação do algoritmo foi trocada.

**Resultado dos modelos principais**

RMSE e MAE estão em R$/L; quanto menores, melhor.

| Modelo | MachineLearning-MVP: RMSE | tcc-mvp: RMSE | MachineLearning-MVP: MAE | tcc-mvp: MAE |
| --- | --- | --- | --- | --- |
| ARIMA | 0,081445 | 0,081453 | 0,027092 | 0,027064 |
| VS-ePL-KRLS | 3,032350 | 0,093818 | 2,823785 | 0,033596 |
| Persistência | 0,095630 | 0,095630 | 0,032596 | 0,032596 |

Para o **VS-ePL-KRLS**, a vantagem é claramente do **tcc-mvp**: redução de 96,91% no RMSE, passando de aproximadamente R$ 3,03/L para R$ 0,094/L. O resultado anterior de erro da ordem de vários reais não se repete na versão nova.

Isso ainda não torna o VS-ePL-KRLS o melhor candidato. No segundo projeto, seu RMSE permanece acima do ARIMA. Ele melhora aproximadamente 1,90% sobre a persistência em RMSE, mas tem MAE maior: 0,033596 contra 0,032596. A decisão registrada de mantê-lo como candidato monitorado, sem promoção, é coerente com esses resultados.

Para o **ARIMA**, há **empate prático**. O primeiro tem RMSE cerca de 0,00945% menor; o segundo tem MAE ligeiramente menor. A diferença de RMSE é somente R$ 0,00000770/L. Isso não sustenta uma preferência operacional baseada em precisão. A acurácia direcional é igual: 59,15% em ambos, considerando apenas semanas em que o preço efetivamente mudou.

Os dois ARIMAs não geraram todas as previsões históricas exatamente iguais: a diferença absoluta média entre suas previsões é R$ 0,00007441/L, e a maior diferença é R$ 0,00244281/L. Ainda assim, o efeito nas métricas agregadas é mínimo.

**Outros candidatos do segundo projeto**

| Candidato | RMSE, R$/L | MAE, R$/L | Acerto direcional nas semanas com mudança |
| --- | --- | --- | --- |
| Paridade de diesel | 0,080807 | 0,027521 | 71,83% |
| ARIMA do primeiro projeto | 0,081445 | 0,027092 | 59,15% |
| ARIMA do segundo projeto | 0,081453 | 0,027064 | 59,15% |
| Ensemble do segundo projeto | 0,084213 | 0,028525 | 78,87% |
| VS-ePL-KRLS do segundo projeto | 0,093818 | 0,033596 | 63,38% |

O modelo de paridade tem o menor RMSE entre esses candidatos: aproximadamente 0,78% abaixo do ARIMA do primeiro projeto. Entretanto, seu MAE é maior que o dos dois ARIMAs. Logo, não vence em todos os critérios e não há base nesta comparação para declarar superioridade geral.

O ensemble apresenta o maior acerto direcional, mas perde em RMSE e MAE para o ARIMA. Acertar o sentido da mudança e acertar o valor do preço são objetivos diferentes; é necessário definir qual métrica responde ao uso pretendido. A persistência não prevê mudanças e, por construção, tem acerto direcional zero quando a métrica exclui semanas sem alteração; isso não significa que seu erro de preço seja ruim.

O segundo repositório possui um relatório de paridade que recomenda promoção, mas seus registros posteriores de gates e o README indicam **não promover**, mantendo ARIMA como primário. Para identificar o modelo efetivamente adotado, considerei esses registros e a release, em vez de tratar uma recomendação experimental anterior como decisão de produção.

**Por que o VS-ePL-KRLS do segundo projeto melhorou**

A inspeção mostra mudanças importantes em relação ao primeiro:

- O pacote novo é vs_epl_krls. O pacote legado vsepl_krls permanece no segundo repositório para compatibilidade e comparação; não representa o challenger novo.
- O candidato escolhido modela a **variação do preço** e reconstrói o nível somando essa variação ao preço da origem. O primeiro VS semanal tenta prever diretamente o nível.
- As features do candidato novo incluem o preço corrente da origem, além de defasagens. No primeiro, as features de preço começam em t-1. Portanto, os candidatos não receberam exatamente a mesma informação.
- O manifesto registra avaliação de **63 candidatos**, com seleção em três blocos temporais de validação de 52 observações antes da janela final de 104 semanas. O primeiro utiliza uma configuração semanal fixa.
- O novo fluxo controla a escala das entradas e a revelação atrasada dos alvos, além de expor limites e sinais de pressão de regras e dicionários.

Essas diferenças ajudam a explicar por que o resultado representa uma melhoria do sistema completo. A análise não isolou a contribuição de cada mudança. Não seria correto atribuir os 96,91% exclusivamente a uma correção matemática do algoritmo ou dizer que as duas implementações foram comparadas sob condições de treinamento idênticas.

**Qual processo considero mais desenvolvido**

O tcc-mvp possui mecanismos mais completos de seleção temporal, testes de causalidade, separação de desenvolvimento e avaliação, metadados dos artefatos, condições explícitas de promoção e registro de previsões futuras. O código atual também contém janelas nacionais fixadas por data, evitando que a chegada de uma observação desloque automaticamente o período final.

Isso é uma evolução em relação ao primeiro projeto, cuja auditoria encontrou vazamento mensal em horizontes longos, uso de resíduos futuros em intervalos semanais, cobertura desalinhada e seleção de features sobre toda a amostra.

Porém, **não certifiquei o segundo repositório como livre de vazamento**. A leitura realizada foi dirigida aos caminhos e artefatos necessários para a comparação, não uma auditoria completa de todos os experimentos. Há ressalvas explícitas no próprio projeto:

- o holdout nacional já foi consultado mais de uma vez durante o desenvolvimento de candidatos;
- a decisão final de promoção ou fallback também consulta critérios nesse holdout;
- existem poucos resultados prospectivos registrados;
- parte dos relatórios e manifestos foi produzida antes de alterações posteriores no código;
- o intervalo do bundle ARIMA apresenta cobertura registrada de 92,3% para uma faixa nominal de 80%, o que indica uma faixa mais conservadora naquele período, não calibração exata;
- os modelos estaduais e de spread têm alvos diferentes e não podem ser usados para alegar ganho sobre o preço nacional.

Essas limitações impedem tratar o resultado retrospectivo como prova definitiva de generalização. Elas não anulam a comparação aritmética das 104 previsões já registradas.

**Conferência da previsão operacional**

A execução local do primeiro projeto e a release mais recente encontrada no segundo registram a mesma previsão ARIMA: **R$ 6,8820506644/L para 23/08/2026**, usando R$ 6,89/L observados em 16/08/2026. Os intervalos diferem porque os procedimentos de calibração não são os mesmos.

Essa coincidência reforça que o avanço do segundo projeto está principalmente no challenger e no processo ao redor dos modelos. Não há evidência de ganho material na previsão pontual do ARIMA que os dois escolheram para produção. As datas desses arquivos são históricas em relação à data desta análise; não verifiquei uma implantação em funcionamento nem atualizei as fontes.

**Parecer**

Se a pergunta é qual **VS-ePL-KRLS** teve melhor desempenho, a resposta é o do **tcc-mvp**, por ampla margem nos resultados alinhados. Se a pergunta é qual **modelo de produção** prevê melhor, a resposta é **empate prático entre os ARIMAs**. Se o objetivo é escolher uma base técnica para continuar o desenvolvimento, considero o **tcc-mvp mais desenvolvido**, condicionado à preservação das ressalvas metodológicas e à coleta de evidência futura.

Para apresentar os resultados no TCC, a afirmação sustentada pelos artefatos é: “Na mesma janela de 104 semanas, a versão do VS-ePL-KRLS presente no tcc-mvp reduziu o RMSE em 96,91% em relação à versão do MachineLearning-MVP. Apesar dessa melhora, o ARIMA permaneceu superior ao VS-ePL-KRLS e apresentou desempenho praticamente igual nos dois projetos.”

**Arquivos usados e resultados reproduzíveis**

- [Métricas recalculadas nesta comparação](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/reports/06_comparacao_modelos_metricas.csv).
- [Previsões alinhadas das 104 semanas](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/reports/06_comparacao_previsoes_alinhadas.csv).
- [Previsões originais do primeiro projeto](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/results/walkforward_preds_h1.csv).
- [Previsões de avaliação final do segundo projeto](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_selection/holdout_predictions_h1.csv).
- [Manifesto da seleção do segundo projeto](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_selection/selection_manifest_h1.json).
- [Previsões do candidato de paridade](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_parity/holdout_predictions.csv).
- [Gates de promoção](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_gates/report.md).
- [Release de produção do segundo projeto](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/reports/vs_epl_krls/s10_product/latest_release.json).
- [Seleção e avaliação do VS-ePL-KRLS novo](C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp/src/vs_epl_krls/selection.py:417).
