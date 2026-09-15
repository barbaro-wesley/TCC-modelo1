# Treinamento e API independentes

O job coleta dados, avalia os candidatos e publica o resultado no PostgreSQL.
A API consulta o banco e usa Redis para rate limit. Os processos podem rodar em
hosts diferentes: não precisam compartilhar disco, ambiente Python ou credenciais.

## Atualizar uma instalação existente

1. Faça backup do banco e preserve os antigos arquivos de resultados.
2. No ambiente administrativo da API, configure a conexão de migração
   `S10_MIGRATION_DATABASE_URL` e execute
   `python -m alembic -c api/alembic.ini upgrade head`.
   A revisão `0002_model_publication` acrescenta cinco tabelas, sem alterar os dados comerciais.
3. Configure o publicador com `.env.training` e instale `training/requirements.txt`.
4. Execute `python scripts/05_atualizacao_semanal.py --forcar --pular-lstm`.
   Confira o status e a origem publicada no banco.
5. Atualize a API e verifique `/health/ready`, login e `/api/v1/forecast`.

Os JSONs antigos não são importados automaticamente: podem conter resultados do
protocolo anterior à correção temporal. Nenhum arquivo legado é apagado.
O histórico de previsões no banco começa na primeira publicação desta versão;
a série observada completa é gravada a cada publicação. Antes da primeira,
forecast/history respondem 503, sem consumo de cota. API antiga pode continuar
servindo seus arquivos durante o preparo, mas eles não serão mais atualizados.

## Treinamento

Na raiz do checkout:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r training/requirements.txt
cp .env.training.example .env.training
# Edite a URL do publicador e S10_CODE_VERSION com o SHA implantado.
chmod 600 .env.training
.venv/bin/python scripts/05_atualizacao_semanal.py --pular-lstm
```

Use uma URL PostgreSQL exclusiva em `S10_TRAINING_DATABASE_URL`.
Em produção, defina `S10_ENVIRONMENT=production` e TLS (`sslmode=require`
ou verificação de certificado). Variáveis do processo prevalecem sobre o arquivo.
O treinamento não carrega `.env`, JWT, Redis nem os pacotes da API.

Na VPS, `deploy/instalar_vps.sh --sem-torch --cron` instala o ambiente e agenda
09:30 e 15:30 (America/Sao_Paulo). Configure o banco/migração antes de executá-lo.
Use `--sem-primeira-execucao` para instalar sem iniciar o job.
LightGBM/XGBoost precisam de OpenMP (`libgomp1` no Debian/Ubuntu).
LSTM é opcional com `--pular-lstm`; as demais avaliações continuam.

Container independente:

```bash
docker build -f training/Dockerfile -t s10-training .
docker run --rm --env-file .env.training s10-training
```

O container usa seu próprio disco de trabalho. Sem persistência, baixa e reconstrói
os dados novamente; a previsão publicada continua no banco. Para cron na VPS,
`deploy/run_semanal.sh` grava logs locais. Systemd é alternativa ao cron:
`deploy/diesel-semanal.service` e `deploy/diesel-semanal.timer`.

## API

Siga [api/README.md](../api/README.md) para criar o administrador e configurar a API.
Seu ambiente instala somente `api/requirements.txt`. A imagem inclui `forecast_store`,
sem pandas, modelos ou bibliotecas de treino.

```bash
docker build -f api/Dockerfile -t s10-api .
docker run --rm --env-file .env -p 127.0.0.1:8080:8080 s10-api
```

As URLs devem ser alcançáveis de cada container; `localhost` dentro dele é o
próprio container. Para desenvolvimento fora de containers, `compose.yaml`
disponibiliza PostgreSQL/Redis no host. Não monte `results/api` na API.

## Permissões do banco

Use três credenciais: proprietário/migrador (DDL), API e publicador.
Crie os usuários pelo provedor ou DBA, sem colocar senhas no repositório.
Depois da migração, ajuste os nomes abaixo aos usuários criados:

```sql
GRANT USAGE ON SCHEMA public TO s10_api, s10_training;
GRANT SELECT ON alembic_version TO s10_api;
GRANT SELECT ON model_runs, model_forecasts, model_observations,
    model_metrics, model_publication TO s10_api;
GRANT SELECT, INSERT, UPDATE ON model_runs TO s10_training;
GRANT SELECT, INSERT ON model_forecasts, model_observations,
    model_metrics TO s10_training;
GRANT SELECT, UPDATE ON model_publication TO s10_training;
GRANT USAGE, SELECT ON SEQUENCE model_runs_id_seq TO s10_training;
```

A API também precisa das permissões de leitura/gravação já usadas nas tabelas
comerciais da revisão 0001. Não dê ao publicador acesso a usuários, senhas,
assinaturas ou consumo. Não conceda escrita nas tabelas `model_*` à API.
Evite usuários com privilégios herdados de proprietário/superusuário. Os GRANTs
acima não revogam privilégios previamente concedidos. A migração não cria usuários
nem concede permissões automaticamente.

## Contrato e consistência

| Tabela | Conteúdo |
| --- | --- |
| `model_runs` | início/fim, estado, versão do código, protocolo, configuração e fontes |
| `model_forecasts` | origem, alvo h=1, modelo, preço, intervalo e metadados JSONB |
| `model_observations` | snapshot da série semanal usada por execução |
| `model_metrics` | métricas por execução, modelo e horizonte |
| `model_publication` | ponteiro único para a última publicação completa |

O job registra `running` antes do download. Não mantém conexão durante o treino.
Previsão, observações, métricas, estado final e ponteiro são escritos numa única
transação curta. Uma falha desfaz a publicação; o job tenta registrar `failed`
com a classe do erro, sem expor mensagens internas pela API.

O lock do arquivo protege o diretório local, inclusive treinos com mais de seis horas.
É liberado pelo sistema quando o processo morre; não apague o arquivo de lock durante
uma execução. Em hosts diferentes, a linha de publicação serializa a confirmação:
uma execução antiga que termina após outra mais nova não substitui a vigente.
Datas de origem não podem retroceder. Repetir uma publicação já confirmada não duplica dados.

Uma nova execução da mesma semana fica preservada para auditoria; `history?series=forecasts`
expõe a última previsão publicada por semana-alvo. O realizado e o erro são calculados
com a série observada da publicação vigente, sem alterar a previsão original.
Datas, filtros, contagem e paginação são processados no SQL.
P10/P90 podem ser nulos enquanto faltam resíduos de calibração.

Sem semana nova e com o mesmo protocolo, o job registra `no_change` e não treina.
Correções de preços na mesma semana exigem `--forcar`.
`--somente-dados` reconstrói arquivos científicos e registra `data_only`, sem
alterar a publicação. Um processo morto abruptamente pode ficar como `running`
no histórico; acompanhe o agendador e logs para detectar execuções abandonadas.

## Operação

- `GET /api/v1/status`: última tentativa e dados da publicação vigente, autenticado.
- `GET /health/ready`: revisão de banco e Redis disponíveis.
- Previsão vencida: forecast/scenarios respondem 503 sem gastar cota; histórico permanece.
- `logs/semanal-*.log`: diagnóstico do treinamento, retenção local de 60 dias.
- Códigos do job: 0 sucesso/sem novidade, 1 falha, 2 sanidade reprovada, 3 diretório ocupado.

CSVs, figuras, relatórios de download e diagnósticos continuam sendo arquivos de
pesquisa internos. `scripts/03_semanal.py` sozinho não publica; seu JSON VS-ePL chama-se
`diagnostico_vsepl_h1.json`. O job 05 é o publicador. O relatório opcional
`scripts/04_producao.py` lê a publicação do banco. Pesos/checkpoints não são gravados
no PostgreSQL nesta alteração; a API serve previsões pré-calculadas.

Não há implantação remota, migração em produção nem agendamento novo automático pelo CI.
