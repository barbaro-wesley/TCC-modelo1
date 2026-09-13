**Relatório de avaliação do modelo e do processo de treinamento**

Projeto: MachineLearning-MVP — previsão do preço médio nacional de revenda do Diesel B S-10.

Data: 11 de setembro de 2026. Base local disponível até 16 de agosto de 2026.

Minha avaliação é que o projeto tem uma estrutura útil para experimentação e já consegue executar o ajuste dos modelos e gerar previsões. Entretanto, sua validação ainda contém falhas que impedem apresentar o sistema como cientificamente validado ou livre de vazamento. O ARIMA é o candidato mais defensável entre os resultados atuais. A implementação avaliada do VS-ePL-KRLS não demonstrou desempenho suficiente para uso preditivo e a LSTM também precisa de investigação antes de integrar uma comparação conclusiva.

Este parecer reúne inspeção do código, auditoria de índices e datas, verificação das métricas salvas e a execução local dos candidatos de produção. A execução local reajustou esses candidatos e gerou uma previsão; não refez o walk-forward, não treinou a LSTM e não atualizou as fontes. Os resultados históricos apresentados são dos artefatos existentes, com novos cálculos descritivos sobre as previsões já salvas. Não foram implementadas correções nesta etapa.

**O que o projeto faz e como aprende**

O objetivo é estimar o preço médio nacional de revenda, e não o preço de um posto específico. Há dois experimentos que precisam permanecer separados na interpretação dos resultados.

O experimento mensal busca reproduzir um artigo com o VS-ePL-KRLS. Utiliza 90 observações entre dezembro de 2012 e maio de 2020. As entradas são os preços de distribuição e revenda do mês de origem; o alvo é o preço de revenda 1, 6 ou 12 meses depois. O treino inicial contém, respectivamente, 72, 66 ou 60 pares. Restam 17, 18 e 18 previsões para teste. A normalização é ajustada somente nas entradas do treino.

O VS-ePL-KRLS aprende incrementalmente. A cada observação, ele pode modificar suas regras, atualizar os centros, ajustar os coeficientes da regressão com kernel e adaptar parâmetros internos. Essa capacidade de atualização é uma característica da implementação; sua existência não comprova que o modelo acompanha bem mudanças no preço. Isso precisa aparecer nas previsões fora da amostra.

O experimento semanal compara oito candidatos. A base contém 703 observações; após defasagens e construção dos alvos, há 690 linhas em h=1, 689 em h=2 e 687 em h=4. São usadas 18 features disponíveis, incluindo defasagens de preços, médias móveis, volatilidade, Brent, câmbio e proxies de reajuste/paridade. As features de ULSD são descartadas por indisponibilidade. O período inicial tem 80 linhas e a avaliação avança cronologicamente, ampliando o histórico de treino.

| Candidato | Procedimento efetivamente executado no walk-forward |
| --- | --- |
| Naive | Repete o último preço conhecido |
| Média móvel | Usa a média dos quatro preços mais recentes |
| ARIMA | Seleciona a ordem por AIC no histórico disponível; reajusta a cada 26 origens e incorpora novas observações entre reajustes |
| ARIMAX | Usa preço e exógenas; reajusta a cada 26 origens |
| LightGBM | 200 árvores, profundidade máxima 4; reajusta a cada 12 origens |
| XGBoost | 200 árvores, profundidade máxima 3; reajusta a cada 12 origens |
| VS-ePL-KRLS | Atualização incremental com atraso do alvo correspondente ao horizonte |
| LSTM | Sequências de 8 entradas, 8 unidades ocultas, 8 épocas por ajuste; reajusta a cada 26 origens |

Os parâmetros desses candidatos estão definidos no código. Não há um processo de busca e seleção de hiperparâmetros com validação temporal interna. Portanto, o ranking compara essas configurações específicas, não o melhor desempenho possível de cada família de modelos.

Em produção, o pipeline escolhe o menor RMSE histórico em h=1 entre os candidatos elegíveis e reajusta os modelos para produzir a previsão. A LSTM não participa dessa etapa. O modelo selecionado nos artefatos locais é o ARIMA, que utiliza a série de preços; ele não utiliza as exógenas empregadas por outros candidatos.

**Minha avaliação do desempenho**

Os resultados históricos de uma semana à frente, em 610 previsões, são:

| Modelo | RMSE, R$/L | MAE, R$/L | Avaliação no experimento atual |
| --- | --- | --- | --- |
| ARIMA | 0,0726 | 0,0279 | Melhor resultado pontual entre os candidatos |
| ARIMAX | 0,0749 | 0,0294 | Próximo do ARIMA, sem ganho agregado sobre ele |
| Naive | 0,0786 | 0,0307 | Referência simples e competitiva |
| Média móvel | 0,1388 | 0,0674 | Inferior ao último preço conhecido |
| XGBoost | 0,2396 | 0,1352 | Não demonstrou benefício nesta configuração |
| LightGBM | 0,2478 | 0,1423 | Não demonstrou benefício nesta configuração |
| VS-ePL-KRLS | 3,4119 | 3,2398 | Erros incompatíveis com uma previsão útil neste experimento |
| LSTM | 4,0921 | 3,8329 | Precisa de revisão do treinamento e das previsões |

O MAE do ARIMA representa aproximadamente 2,8 centavos por litro de erro absoluto médio no conjunto avaliado. O RMSE, que dá mais peso a erros grandes, é de aproximadamente 7,3 centavos por litro. Esses números não significam que cada previsão terá erro inferior a esses valores.

A redução de RMSE do ARIMA em relação ao naive é de 7,63%; a redução de MAE é de 9,20%. É uma melhora observada, mas moderada. O próprio teste Diebold–Mariano implementado no projeto registra p=0,0522 para ARIMA contra naive em h=1, acima do limiar de 0,05. Esse resultado não estabelece igualdade entre os modelos, nem uma superioridade definitiva do ARIMA. A seleção pelo mesmo histórico e as falhas identificadas exigem uma avaliação posterior independente.

Recalculei também o RMSE em janelas recentes, usando as previsões já armazenadas:

| Modelo | Todas as 610 previsões | Últimas 156 previsões | Últimas 52 previsões |
| --- | --- | --- | --- |
| ARIMA | 0,0726 | 0,0682 | 0,1079 |
| ARIMAX | 0,0749 | 0,0705 | 0,1108 |
| Naive | 0,0786 | 0,0800 | 0,1287 |
| VS-ePL-KRLS | 3,4119 | 3,1562 | 3,2469 |
| LSTM | 4,0921 | 5,4728 | 5,8101 |

As últimas 156 previsões têm origens entre 20/08/2023 e 09/08/2026; as últimas 52, entre 17/08/2025 e 09/08/2026. Essas janelas foram examinadas depois de conhecer os resultados e não constituem um teste reservado independente.

O ARIMA continua melhor que o naive nesses recortes, mas seu erro nas últimas 52 previsões é maior que a média histórica. Isso reforça a necessidade de acompanhar períodos recentes, em vez de usar apenas um RMSE agregado sobre mais de uma década.

O VS-ePL-KRLS apresenta RMSE aproximadamente 47 vezes maior que o ARIMA em h=1. Seu erro médio com sinal, calculado como realizado menos previsto, é aproximadamente +3,24 R$/L, praticamente igual ao MAE. A LSTM apresenta o mesmo padrão, com aproximadamente +3,83 R$/L. Os resultados indicam forte subestimação, e não apenas algumas previsões extremas isoladas.

Nas três últimas origens salvas, a LSTM prevê aproximadamente R$ 0,7953/L, enquanto os alvos estão entre R$ 6,89/L e R$ 6,94/L. É um sinal concreto para investigar ajuste, escala, convergência e construção das entradas. Esta análise não isolou qual desses fatores causa o problema. Também não permite concluir que LSTM ou VS-ePL-KRLS sejam inadequados em geral: o resultado se refere às implementações e configurações deste repositório.

**O que considero bem feito**

O projeto separa coleta, preparação, modelos, avaliação e publicação. Essa organização facilita corrigir problemas sem reconstruir o sistema inteiro. A presença de naive, média móvel, modelos estatísticos, árvores e modelos incrementais oferece referências úteis para avaliar se a complexidade adicional compensa.

A validação semanal avança no tempo, sem divisão aleatória. As defasagens e médias móveis usam o passado, a normalização mensal é ajustada somente no treino e a normalização semanal do VS-ePL-KRLS fica congelada no período inicial. No caminho semanal principal, os cortes dos alvos respeitam o horizonte nas datas locais verificadas, sob a hipótese de que cada semana de origem já esteja publicada.

Também considero correta a decisão operacional de escolher o ARIMA quando o modelo de interesse acadêmico perde. O relatório mensal reconhece que a reprodução não foi alcançada. Essa transparência precisa ser mantida na apresentação do trabalho.

**O que reduz minha confiança no treinamento e na avaliação**

| Problema | Evidência e consequência |
| --- | --- |
| Vazamento no treino mensal h=6 e h=12 | A primeira previsão de junho/2018 em h=6 já usa novembro/2018 no treino; em h=12, a origem dezembro/2017 também já recebeu novembro/2018. Os alvos não aguardam sua realização. |
| Vazamento na calibração semanal h=2 e h=4 | Os intervalos usam resíduos de previsões cujo preço real ainda não seria conhecido na origem. Isso afeta intervalos e sua avaliação. |
| Seleção de features com informação futura | A regra de disponibilidade acima de 80% consulta a série inteira. Alterar ausências no fim da base muda as entradas usadas no começo. |
| Cobertura com vetores desalinhados | O código filtra os valores reais, mas não aplica a mesma máscara aos intervalos. A cobertura divulgada fica incorreta. |
| Horizonte calculado por linhas | O alvo de “uma semana” liga 16/08/2020 a 18/10/2020. As lacunas alteram o significado do experimento. |
| Importância calculada sobre dados de treino | O último LightGBM é avaliado sobre um bloco no qual 600 das 610 linhas já pertenceram ao seu treino. A importância não representa uma avaliação independente. |
| Ausência de teste final independente | O mesmo histórico define o vencedor e fornece o erro apresentado para justificar a escolha. |
| Datas de publicação não modeladas | O código usa o início da semana como data, mas precisa demonstrar que o preço agregado já estava disponível no instante da previsão. |

O vazamento mensal é grave porque afeta diretamente o aprendizado. O vazamento de resíduos afeta outra etapa: a calibração da incerteza. Não encontrei evidência de que esses dois problemas tenham introduzido alvos futuros no ajuste pontual do ARIMA semanal. Isso não torna a avaliação completa válida, pois permanecem problemas de calendário, disponibilidade e seleção.

Há ainda uma desigualdade informacional na comparação. O ARIMA e o naive usam o preço da origem t, mas as features de preços dos modelos supervisionados começam em revenda_l1, isto é, t-1, enquanto o alvo está em t+h. Se o preço t está disponível para um candidato, ele também deveria estar disponível aos demais em uma comparação com a mesma informação. Se não está publicado, nenhum candidato deve usá-lo. Essa assimetria pode prejudicar os modelos supervisionados; seu efeito ainda não foi medido e não explica, por si só, a magnitude dos erros observados.

A LSTM também usa uma janela de inferência atrasada em h>1. O ARIMAX de produção utiliza um painel que descarta a última origem e acaba prevendo a semana anterior à data anunciada no payload. São erros de alinhamento que precisam ser corrigidos antes de usar esses candidatos como evidência comparativa.

**Minha avaliação dos intervalos e das probabilidades**

Não considero os intervalos atuais suficientemente validados para serem apresentados como uma faixa confiável de 80%. Corrigindo somente o alinhamento dos vetores, a cobertura histórica do ARIMA h=1 passa de 0,17% para 66,10%. Essa frequência está abaixo dos 80% nominais do intervalo P10–P90. É uma correção do cálculo sobre previsões existentes, não o resultado de uma nova calibração.

Além disso, o backtest usa quantis de um histórico crescente de resíduos, enquanto a produção usa somente os últimos 80 resíduos. Logo, mesmo a cobertura histórica recalculada não valida exatamente a regra aplicada à previsão publicada.

As probabilidades de alta, estabilidade e queda são frequências obtidas ao combinar a previsão com resíduos históricos. Não há, no material avaliado, uma verificação específica de calibração dessas probabilidades. Eu as apresentaria como estimativas experimentais, explicitando o método e o período dos resíduos, até avaliar sua qualidade fora da amostra.

**Execução realizada e o que ela comprova**

O ajuste dos candidatos de produção terminou com sucesso no ambiente local e gerou uma previsão finita. O ARIMA foi selecionado pelo ranking previamente existente e previu R$ 6,8821/L para a semana de 23/08/2026, a partir da base encerrada em 16/08/2026.

Como a execução ocorreu em 11/09/2026, esse resultado é uma execução sobre uma base histórica desatualizada, não uma previsão da próxima semana em relação à data atual. Ela comprova que o caminho local de ajuste e geração do JSON funciona. Não comprova atualização das fontes, funcionamento do agendamento em produção, reprodução de todo o backtest ou correção das falhas metodológicas.

Os 10 testes existentes passaram na auditoria anterior, com um aviso de permissão de gravação do cache. A suíte não verifica os vazamentos, as lacunas de calendário e os alinhamentos descritos aqui. Portanto, passar nos testes atuais não é suficiente para atestar a validade do experimento.

**Prioridades e critérios para considerar o processo confiável**

| Prioridade | Ação | Evidência esperada após a correção |
| --- | --- | --- |
| 1 | Definir origem, alvo e data de disponibilidade de cada observação | Todo alvo de treino e todo resíduo de calibração disponível antes ou no instante de emissão |
| 1 | Corrigir treinamento mensal, atraso dos resíduos, máscaras e calendário | Testes de causalidade e de alinhamento passando para todos os horizontes |
| 2 | Fixar ou selecionar features somente no treino e igualar informação disponível entre candidatos | Alterações em dados futuros não mudam entradas, previsões ou intervalos anteriores |
| 2 | Corrigir inferência da LSTM e produção do ARIMAX | A previsão de cada candidato corresponde exatamente à data anunciada |
| 2 | Reavaliar incerteza usando a mesma regra de resíduos da produção | Cobertura e largura medidas em blocos posteriores, sem calibração com futuro |
| 3 | Separar escolha do modelo e teste final | Comparação com naive em um período não usado para escolher configuração ou vencedor |
| 3 | Investigar VS-ePL-KRLS e LSTM antes de ampliar a busca de parâmetros | Previsões com escala coerente, erros diagnosticados e aprendizado verificado em casos controlados |
| 3 | Registrar versão dos dados, parâmetros, dependências e código | Uma execução histórica pode ser reproduzida com seus insumos originais |

Depois dessas correções, o próximo experimento deve refazer o walk-forward e gerar novamente métricas e relatórios. Para comparar candidatos, eu manteria o naive como referência obrigatória, usaria o ARIMA como principal candidato atual e permitiria que a nova avaliação determinasse o vencedor. Não há motivo, nos resultados presentes, para selecionar o VS-ePL-KRLS apenas por ser o modelo central do estudo.

**Parecer para apresentação acadêmica e uso do sistema**

Para um TCC, o projeto pode sustentar um estudo sobre reprodução, adaptação temporal e comparação de métodos, inclusive com resultado negativo para o modelo proposto. A formulação mais defensável neste momento é: “Na configuração avaliada, o ARIMA apresentou o menor erro pontual, enquanto a adaptação do VS-ePL-KRLS não demonstrou vantagem; a auditoria identificou falhas metodológicas que exigem correção e reavaliação.”

Eu não apresentaria a reprodução mensal h=6 e h=12 como válida, nem afirmaria superioridade geral de uma família de modelos, probabilidades calibradas ou treinamento integralmente sem vazamento. Também não usaria o RMSE histórico para prometer o mesmo erro em preços individuais de postos.

Meu parecer é favorável à continuidade do projeto como protótipo de pesquisa. Para disponibilizar previsões como resultado validado, condicionaria o avanço à correção do protocolo temporal, à reavaliação independente e à verificação de atualização dos dados. O investimento mais útil agora está na qualidade do experimento e no diagnóstico dos candidatos, antes de aumentar a complexidade dos modelos.

**Fontes locais e rastreabilidade**

- [Auditoria detalhada de treinamento e vazamento](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/reports/04_auditoria_treinamento.md).
- [Resultados históricos dos modelos](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/results/semanal_benchmarks.csv).
- [Previsões históricas de uma semana usadas nos recálculos](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/results/walkforward_preds_h1.csv).
- [Resultado da execução local](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/results/previsao_execucao_local.json).
- [Treinamento mensal](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/scripts/02_reproducao.py:33) e [treinamento semanal](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/scripts/03_semanal.py:68).
- [Preparação dos painéis](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/src/data/panel.py:40), [calibração dos intervalos](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/src/eval/intervals.py:8) e [previsão de produção](C:/Users/Wesley.Barbaro/Documents/TCC-MODELO-PABLO/MachineLearning-MVP/src/pipeline/forecast.py:129).
