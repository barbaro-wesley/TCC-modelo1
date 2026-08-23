# Previsão do Diesel B S-10 (Brasil)

Três blocos separados:

1. **Reprodução do artigo** (mensal, dez/2012–mai/2020) — **não reproduzido** no critério de ±10% (melhor RMSE 0,077 vs 0,060).
2. **Adaptação semanal** walk-forward (h = 1, 2, 4 semanas).
3. **Produção:** vencedor do walk-forward em h=1 (hoje ARIMA) para a próxima semana, republicado toda semana pelo job automático.

O VS-ePL-KRLS do artigo **não foi selecionado** para produção semanal (RMSE 3,42 vs 0,073 do ARIMA).

## Previsão atual (próxima semana)

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

Atualizado automaticamente em 2026-08-23T01:02:38+00:00. Arquivos: `results/previsao_proxima_semana.json`, `results/api/`.

<!-- AUTO:PREVISAO:FIM -->

Esta seção é reescrita pelo job semanal — não edite à mão entre os marcadores.

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

`results/api/` é o contrato de leitura (JSON estável, reescrito a cada execução):

| Arquivo | Conteúdo |
| --- | --- |
| `api/previsao.json` | previsão da próxima semana, modelo vencedor, P10/P90, previsão de cada modelo |
| `api/historico.json` | série semanal completa da ANP + previsões passadas com erro realizado |
| `api/status.json` | saúde do pipeline: última execução, status, última semana da ANP, fontes |

Estado interno em `results/pipeline_state.json`; logs em `logs/`.

## Como rodar

No macOS, LightGBM/XGBoost precisam de OpenMP: `brew install libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
