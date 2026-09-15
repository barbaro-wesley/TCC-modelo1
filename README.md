# Previsão do Diesel B S-10 (Brasil)

> Correção temporal: o código agora usa o protocolo `calendar-mature-labels-v1`.
> As métricas e a previsão histórica abaixo ainda não foram recalculadas com ele.
> Reexecute os experimentos antes de usá-las como validação do código corrigido.
> Escopo, testes e limitações: [correções temporais](reports/08_correcoes_temporais.md).

Três blocos separados:

1. **Reprodução do artigo** (mensal, dez/2012–mai/2020) — **não reproduzido** no critério de ±10% (melhor RMSE 0,077 vs 0,060).
2. **Adaptação semanal** walk-forward (h = 1, 2, 4 semanas).
3. **Produção:** vencedor do walk-forward em h=1 (hoje ARIMA) para a próxima semana, republicado toda semana pelo job automático.

O VS-ePL-KRLS do artigo **não foi selecionado** para produção semanal (RMSE 3,42 vs 0,073 do ARIMA).

## Registro histórico de previsão (anterior à correção temporal)

<!-- AUTO:PREVISAO:INICIO -->

Última semana ANP: **2026-08-16**, preço **R$ 6,89/L**.

Previsão para a semana de **2026-08-23**, modelo **ARIMA** (menor RMSE walk-forward em h=1):

| | R$/L |
| --- | --- |
| ARIMA (produção) | **6,88** (6,85 – 6,90) |
| naive | 6,89 |
| media_movel | 6,92 |
| ARIMAX | 6,90 |
| LightGBM | 6,86 |
| XGBoost | 6,84 |
| VS-ePL-KRLS | 4,98 |

Prob. alta / estável / queda (±0,02): 8% / 60% / 32%.

Registro de 2026-08-23T01:02:38+00:00, preservado como evidência histórica.

<!-- AUTO:PREVISAO:FIM -->

A previsão vigente agora é consultada pela API, a partir do PostgreSQL. O treinamento não altera o README.

## Atualização semanal automática

O job baixa os dados novos da ANP, refaz as features, roda o walk-forward de novo
e publica a previsão do modelo vencedor. Sem semana nova ele sai em segundos, então
pode rodar todo dia sem saber o dia exato em que a ANP publica.

```bash
python scripts/05_atualizacao_semanal.py              # só retreina se houver semana nova
python scripts/05_atualizacao_semanal.py --forcar     # retreina de qualquer jeito
python scripts/05_atualizacao_semanal.py --pular-lstm # dispensa o torch
```

Instalação do cron na VPS: veja [`deploy/README.md`](deploy/README.md).

### Saídas para a API

O contrato é o PostgreSQL, compartilhado entre processos independentes:

| Componente | Responsabilidade |
| --- | --- |
| `training/` + `src/` | download, features, treino, avaliação e publicação transacional |
| `forecast_store/` | esquema, gravação e consultas dos resultados |
| `api/` | HTTP, autenticação, assinaturas, cotas e leitura das publicações |
| PostgreSQL | execuções, previsões, métricas, observações e publicação vigente |
| Redis | rate limit da API |

Previsão, métricas e série observada são confirmadas juntas. Falhas preservam a
publicação anterior; histórico e estado deixam de depender de arquivos JSON.
Metadados flexíveis ficam em colunas JSONB; preços e datas têm colunas próprias.
Os arquivos científicos continuam locais ao treinamento. Não há volume compartilhado
com a API nem treinamento durante requisições. Veja [operação e migração](deploy/README.md).

A API Python/FastAPI em [`api/`](api/README.md) entrega esses resultados com autenticação, empresas, usuários, planos em BRL, assinaturas, cotas mensais no PostgreSQL e rate limit distribuído no Redis. Todas as listagens são paginadas e têm limite global de tamanho e profundidade. As rotas antigas foram substituídas por `/api/v1/forecast`, `/api/v1/history` e `/api/v1/status`.

O backend tem ambiente e dependências próprios; não instala bibliotecas de treinamento para servir HTTP. Consulte [`api/README.md`](api/README.md) para configuração Neon, migrações, testes e criação do administrador. O pipeline científico e os relatórios permanecem separados. A cópia experimental em `reports/auditoria_nacional_h1/implementacao/` é evidência histórica, não é importada nem publicada pela API.

## Como rodar

No macOS, LightGBM/XGBoost precisam de OpenMP: `brew install libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
cp .env.training.example .env.training
# Configure o banco e aplique Alembic conforme api/README.md antes do job semanal.
PYTHONPATH=src python scripts/01_download.py
PYTHONPATH=src python scripts/02_reproducao.py
PYTHONPATH=src python -u scripts/03_semanal.py
PYTHONPATH=src python -u scripts/05_atualizacao_semanal.py --forcar
```

## Dados

- ANP SHLP mensal/semanal Brasil (S-10)
- IPEADATA Brent (`EIA366_PBRENT366`) e câmbio (`GM366_ERC366`)
- Stooq ULSD: indisponível — o site responde com um desafio de bot em vez do CSV, então as features `ulsd_*` ficam vazias e o painel as descarta

Distribuição ANP só até 17/08/2020. Buraco de pesquisa 18/08/2020–17/10/2020.
