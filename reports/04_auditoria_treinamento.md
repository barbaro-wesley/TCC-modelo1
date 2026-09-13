**Auditoria do treinamento e da avaliação — 11/09/2026**

O repositório contém vazamentos confirmados, mas eles não estão distribuídos igualmente pelo pipeline. A reprodução mensal em h=6 e h=12 usa alvos futuros no treinamento. A avaliação semanal usa resíduos futuros na calibração de intervalos em h=2 e h=4. A seleção das features consulta a disponibilidade de dados da amostra inteira. No treinamento semanal principal, os cortes de alvos respeitam o horizonte nos dados locais, assumindo que a observação da semana de origem já foi publicada.

Esta análise examinou código, dados processados, previsões e relatórios locais. Não reexecutou todo o treinamento nem baixou novas versões das fontes. Os números de desempenho abaixo são dos resultados salvos; as verificações de índices, cobertura e causalidade foram executadas nesta auditoria. Somente este relatório foi acrescentado; o código e os resultados existentes não foram corrigidos.

**Como o treinamento funciona**

O alvo é o preço médio nacional de revenda do Diesel B S-10. O fluxo baixa séries ANP e séries de Brent e câmbio, monta dados processados e compara vários modelos. A API em Go serve os arquivos produzidos pelo pipeline Python; ela não treina modelos.

Na reprodução mensal, os dados locais contêm 90 observações, de dezembro/2012 a maio/2020. Cada entrada contém distribuição(t) e revenda(t); o alvo é revenda(t+h). A divisão é cronológica:

| Horizonte mensal | Pares disponíveis | Treino inicial | Teste |
| --- | --- | --- | --- |
| 1 | 89 | 72 | 17 |
| 6 | 84 | 66 | 18 |
| 12 | 78 | 60 | 18 |

O MinMaxScaler é ajustado somente nas entradas do treino. O VS-ePL-KRLS aprende uma observação por vez: ajusta regras e seus centros, coeficientes da regressão com kernel, larguras dos kernels e o parâmetro beta. Há variantes com e sem passo variável e duas convenções de limiares. Os parâmetros estão fixados no código, com valores mensais atribuídos ao artigo; não há busca automática de hiperparâmetros. No teste mensal, prevê uma linha e imediatamente atualiza com seu alvo. Essa atualização precisa ser atrasada para horizontes maiores que um.

Referências: scripts/02_reproducao.py:33 e :51; src/vsepl_krls/paper.py:36 e :46; src/vsepl_krls/model.py:290. A fidelidade matemática ao artigo não foi objeto desta auditoria. O relatório mensal declara que a reprodução não foi alcançada.

Na adaptação semanal, a base local tem 703 observações, de 30/12/2012 a 16/08/2026. Há 18 features efetivamente selecionadas: preços defasados, médias móveis, volatilidades, Brent, câmbio, Brent em reais e proxies de reajuste/paridade. As duas features ULSD são descartadas por falta de dados. Distribuição não integra FEATURE_COLS.

Os painéis h=1, h=2 e h=4 têm 690, 689 e 687 linhas. As primeiras 80 linhas formam o período inicial; a avaliação começa na origem 05/10/2014. O painel faz y[t] = revenda[t+h]. A validação é walk-forward com janela crescente, sem embaralhamento.

| Modelo | Ajuste efetivamente executado |
| --- | --- |
| VS-ePL-KRLS | Atualização incremental com o par t-h; normalização congelada nas primeiras 80 entradas; até 8 regras e 15 elementos por dicionário |
| Naive | Repete o último preço conhecido |
| Média móvel | Média dos quatro últimos preços disponíveis |
| ARIMA | Seleciona ordem por AIC no histórico disponível; reajusta a cada 26 origens e incorpora a observação nova entre reajustes |
| ARIMAX | Escolhe entre três ordens por AIC; reajusta a cada 26 origens; repete a última exógena disponível para o futuro |
| LightGBM | 200 árvores, profundidade máxima 4, 15 folhas; reajuste a cada 12 origens |
| XGBoost | 200 árvores, profundidade máxima 3; reajuste a cada 12 origens |
| LSTM | Sequências de 8 entradas, 8 unidades ocultas, 8 épocas, Adam e MSE; reajuste a cada 26 origens, em subprocesso |

Essas periodicidades são as passadas pelo main, não os valores padrão das funções. O texto gerado para o relatório semanal fala em 4 e 8 semanas e está desatualizado. Referência: scripts/03_semanal.py:208.

O job semanal baixa novamente os dados, reconstrói features e reexecuta a comparação quando há semana nova ou execução forçada. Depois escolhe o menor RMSE em h=1 entre candidatos elegíveis e ajusta os modelos para a previsão final. O vencedor salvo é ARIMA: RMSE 0,072558 R$/L, contra 0,078554 do naive e 3,411888 do VS-ePL-KRLS, em 610 previsões h=1. O ARIMA usa a série de preços; as exógenas mencionadas no model card pertencem a outros candidatos.

Referências: src/pipeline/weekly.py:323; src/pipeline/forecast.py:61 e :105; results/semanal_benchmarks.csv.

**Vazamentos confirmados e seu alcance**

1. **Alta prioridade — alvos futuros no treinamento mensal h=6 e h=12.** O treino inicial usa todos os ytr e o teste incorpora yte[i] imediatamente após prever. Mas yte[i] só se realiza h meses depois da origem. Prever antes de atualizar a mesma linha não impede que as próximas previsões usem esse futuro. Em h=6, a primeira origem de teste é junho/2018 e o treino já contém o preço de novembro/2018: cinco meses adiante. Em h=12, a primeira origem é dezembro/2017 e o treino contém novembro/2018: onze meses adiante. Confirmei esses índices executando a função run_one extraída do código com um modelo instrumentado que registra os alvos recebidos. Em h=1, o alvo mais recente do treino coincide com o mês de origem; esse vazamento por horizonte não aparece, sob a hipótese de que o mês já está publicado. Corrigir o corte inicial e liberar cada atualização somente quando a data do alvo for conhecida. Referência: scripts/02_reproducao.py:33.

2. **Alta prioridade — resíduos futuros nos intervalos semanais h=2 e h=4.** conformal_p10_p90 usa residuals[:t] sem receber o horizonte. O resíduo da origem t-1 depende do preço t-1+h, ainda desconhecido em t quando h>1. Uma prova por perturbação confirmou o efeito: em h=4 e origem 25, mudar somente os resíduos ainda indisponíveis das origens 22, 23 e 24 alterou o limite superior de 0 para 60. Isso contamina a calibração e sua avaliação, mas não demonstra vazamento no ajuste das previsões pontuais. Em uma grade regular, o corte seria até t-h inclusive; com falhas de calendário, deve usar a data real de disponibilidade. Referências: src/eval/intervals.py:19; scripts/03_semanal.py:165.

3. **Prioridade média — seleção de features consulta o futuro.** load_panel inclui colunas com mais de 80% de valores presentes em toda a série, antes do walk-forward. A representação usada no passado depende, portanto, da disponibilidade futura. Confirmei alterando apenas brent_l1 nas últimas 200 linhas: a coluna deixou de integrar o painel inteiro, inclusive seu treino inicial. Isso é contaminação do pré-processamento por informação futura, sem uso direto do valor do alvo. Não quantifiquei o efeito nos erros salvos. Fixar a lista por decisão prévia ou selecioná-la apenas no treino de cada janela. Referência: src/data/panel.py:48.

**Outros problemas que comprometem a interpretação dos resultados**

- **Cobertura com datas desalinhadas.** eval_model filtra y, mas passa lo e hi completos para coverage; a função trunca os vetores por tamanho. Assim compara preços de datas posteriores com intervalos de datas anteriores e inclui posições NaN. Recalculei a cobertura do ARIMA aplicando a mesma máscara aos três vetores: h=1 passa de 0,1695% para 66,1017%; h=2, de 0,6791% para 64,1766%; h=4, de 3,9182% para 63,3731%. Os valores de h=2 e h=4 ainda usam a calibração contaminada descrita acima; não são resultados de um pipeline integralmente corrigido. O intervalo P10–P90 tem faixa nominal de 80%, acima da cobertura observada em h=1. Referências: scripts/03_semanal.py:166; src/eval/metrics.py:51.

- **Horizonte por posição de linha, não por calendário.** shift(-horizon) liga 16/08/2020 a 18/10/2020 como h=1, um intervalo de nove semanas. Liga também 09/08/2015 a 23/08/2015. Nos painéis locais, há 2, 4 e 8 pares com diferença de datas incompatível com h=1, 2 e 4 semanas, respectivamente. O filtro do buraco ANP só exclui alvos dentro do intervalo vazio; não impede atravessá-lo. Defasagens e médias também contam observações. Corrigir com calendário explícito e regras para pares que atravessam lacunas. Referência: src/data/panel.py:43.

- **Importância de features avaliada em dados já usados pelo modelo.** O último LightGBM treinado é aplicado a todo X[80:] para permutation importance. Em h=1, 600 das 610 linhas dessa avaliação pertencem ao treino desse último modelo. Não é uma explicação fora da amostra das previsões walk-forward e pode favorecer efeitos aprendidos no treino. Calcular importância em blocos futuros relativos a cada ajuste ou em uma reserva temporal final. Referência: scripts/03_semanal.py:291.

- **O vencedor é escolhido na mesma avaliação cujo erro é divulgado.** O ranking completo define o modelo final; não há um teste final independente dessa escolha. Selecionar assim um modelo para a próxima previsão é operacionalmente possível, mas seu menor RMSE histórico não é uma estimativa independente do desempenho da seleção. Usar validação temporal para escolher e uma reserva posterior para avaliar, ou simular a seleção de modelos em cada origem usando somente erros já realizados. Referência: src/pipeline/forecast.py:99.

- **Disponibilidade real dos dados não está modelada.** parse_weekly usa DATA INICIAL como data, embora o preço agregado se refira à semana. O backtest usa revenda[t] como conhecida na origem t, e o download substitui séries históricas pela versão mais recente, mantendo somente uma cópia anterior. Se t significa começo da semana, há antecipação do preço da própria semana; se significa emissão após fechamento/publicação, a hipótese pode ser válida. Falta registrar data de emissão e de publicação por observação para provar isso e controlar revisões históricas. Trata-se de risco operacional não quantificado, não de prova de que toda previsão publicada vazou. Referências: src/data/anp.py:114 e :124; src/data/download.py:29; src/pipeline/weekly.py:259.

- **Previsão de produção ARIMAX atrasada.** _prever_arimax usa panel['y_prev']; load_panel descarta a última origem porque seu alvo é desconhecido. Com a base local, o ARIMAX termina em 09/08/2026 e prevê um passo até 16/08, enquanto o payload final é rotulado 23/08. Corrigir usando a última observação publicada e suas exógenas. Isso é erro de alinhamento, não vazamento, e não afeta diretamente o valor publicado do vencedor ARIMA. Referência: src/pipeline/forecast.py:54.

- **Entrada da LSTM atrasada em h>1.** A construção de sequências associa X[t-8:t] a y[t]. Na inferência, o código passa X[:t-h+1], logo a janela corresponde a y[t-h+1], mas a previsão é avaliada contra y[t]. Há defasagem de h-1 posições; h=1 não tem esse deslocamento. Separar o limite de rótulos disponíveis para ajuste da janela de entradas usada para inferência. Referências: src/benchmarks/lstm.py:21; scripts/_lstm_wf.py:30.

- **Funções auxiliares inseguras, fora do fluxo principal.** src/eval/walkforward.py contém uma versão online cujo horizon não controla as atualizações e uma versão batch que usa todos os y[:t]. A primeira pode usar alvos futuros para h>2, considerando que atualiza depois de prever; a segunda para h>1, se receber os alvos deslocados. Não encontrei chamadas a essas funções no treinamento atual. Não atribuir seus erros ao run_vsepl de scripts/03_semanal.py, que tem outra implementação.

**Proteções existentes e verificações realizadas**

O código semanal principal atualiza VS-ePL-KRLS com y[t-h] e limita os modelos supervisionados a y[:t-h+1]. Verifiquei todos os cortes a partir da origem 80 contra as datas reais dos alvos nos três painéis locais; nenhum corte incluiu um alvo com data posterior à origem. Isso não resolve a questão da hora/dia de publicação mencionada acima.

As defasagens e janelas móveis usam o passado; o preenchimento do painel usa ffill; a normalização mensal usa apenas treino; a normalização semanal do VS-ePL-KRLS fica congelada no período inicial. ARIMA seleciona sua ordem com AIC sobre o histórico da origem. Treinar novamente com todo o histórico conhecido para emitir uma previsão futura é válido por si só.

Há chamadas bfill, mas no caminho principal de avaliação as features já passaram por dropna; não identifiquei nelas a causa de vazamento nos resultados atuais. leak_check examina nomes de colunas proibidas e sufixos, sem verificar datas, cortes de alvos, resíduos ou disponibilidade. Passar nessa função não certifica ausência de vazamento.

Executei python -m pytest -q: 10 testes passaram, com um aviso de permissão ao gravar o cache do pytest. A suíte cobre operações do modelo, sanidade, escolha do vencedor e histórico; não cobre os vazamentos e alinhamentos acima. As verificações adicionais da auditoria foram executadas em memória, sem substituir os resultados salvos.

**Ordem sugerida para corrigir e reavaliar**

1. Representar data de origem, data do alvo e disponibilidade; corrigir o atraso de rótulos mensal e a calibração de resíduos por horizonte.
2. Corrigir máscaras da cobertura e pares que atravessam falhas no calendário.
3. Restringir seleção de features ao passado; acrescentar testes em que mudanças no futuro não alteram previsões ou intervalos anteriores.
4. Corrigir alinhamento da LSTM e do ARIMAX de produção; avaliar importância em blocos fora do treino.
5. Separar seleção de modelo da avaliação final e registrar versões dos dados e configuração de cada execução.
6. Reexecutar os experimentos e regenerar relatórios antes de apresentar seus números como validação livre de vazamento.

Os achados não demonstram que o RMSE pontual semanal do ARIMA foi produzido com alvos futuros. Demonstram que o pipeline completo ainda não sustenta uma afirmação geral de treinamento e avaliação sem vazamento, e que várias métricas e comparações precisam ser refeitas.
