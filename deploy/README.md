# Atualização semanal na VPS

O job baixa os dados novos da ANP, refaz as features, roda o walk-forward de novo e
publica a previsão do modelo vencedor em JSON. Tudo fica em arquivo no disco — a API
em Go só precisa ler `results/api/`.

## Instalação

```bash
sudo apt install python3 python3-venv build-essential libgomp1
git clone <repo> /srv/diesel-s10
cd /srv/diesel-s10
./deploy/instalar_vps.sh --sem-torch --cron
```

`libgomp1` é o OpenMP que LightGBM e XGBoost exigem — sem ele o import falha em
imagem enxuta de VPS. Confira também a versão do Python: `requirements.txt` está
pinado em `numpy==2.5.0` / `pandas==3.0.5`, que pedem Python recente (3.12+). Se o
`python3` da VPS for mais antigo e o `pip install` reclamar de wheel, instale um
Python novo (`deadsnakes` no Ubuntu) e aponte com
`PYTHON_BASE=/usr/bin/python3.13 ./deploy/instalar_vps.sh ...`.

`--sem-torch` pula ~800 MB de dependência: o LSTM é só benchmark e perde por ordens de
grandeza no walk-forward. Sem torch, o job precisa rodar sempre com `--pular-lstm`
(o `--cron` já monta as linhas assim). Se quiser o benchmark completo para o TCC,
instale sem a flag e rode `deploy/run_semanal.sh --forcar` de vez em quando.

O script é idempotente: rodar de novo não recria a venv nem duplica linha de cron.

## Agendamento

Padrão instalado pelo `--cron` (veja `crontab.example`):

```
CRON_TZ=America/Sao_Paulo
30 9  * * * /srv/diesel-s10/deploy/run_semanal.sh --pular-lstm
30 15 * * * /srv/diesel-s10/deploy/run_semanal.sh --pular-lstm
```

**Por que duas vezes por dia e não uma vez por semana:** a ANP publica a síntese semanal
no meio da semana, mas a data varia e às vezes atrasa. Quando não há semana nova, o job
compara a última data da planilha com `results/pipeline_state.json` e sai em segundos,
sem treinar nada. Então rodar todo dia custa quase nada e você nunca perde a publicação.

Quem preferir systemd em vez de cron:

```bash
sudo cp deploy/diesel-semanal.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now diesel-semanal.timer
systemctl list-timers diesel-semanal
```

O timer tem `Persistent=true`: se a VPS estiver desligada no horário, ele dispara ao voltar.

## O que uma execução faz

1. Baixa ANP (mensal + semanal), Brent e câmbio do IPEADATA, ULSD do Stooq.
   Cada arquivo vai para `.part`, é validado e só então substitui o anterior — uma
   página de erro ou de captcha nunca sobrescreve dado bom. A versão anterior fica em `.prev`.
2. Lê a planilha semanal e compara a última semana com o estado salvo. Igual → sai (exit 0).
3. Checagens de sanidade: planilha não pode encolher, data não pode estar no futuro,
   preço não pode dar salto implausível. Reprovou → aborta **sem** tocar no processado (exit 2).
4. Reconstrói `data/processed/`.
5. Roda o walk-forward completo (`scripts/03_semanal.py`, h = 1, 2, 4).
6. Reajusta cada candidato na série inteira, escolhe o de menor RMSE em h=1 e publica
   a previsão dele com P10/P90 conformais.
7. Atualiza histórico, `results/api/`, os relatórios e a seção "Previsão atual" do README.

Uma execução com retreino leva alguns minutos; sem semana nova, segundos.

## Códigos de saída

| Código | Significado |
| --- | --- |
| 0 | ok — inclui "sem semana nova" |
| 1 | erro (rede, parsing, treino) |
| 2 | dados novos reprovados na sanidade; o processado anterior foi mantido |
| 3 | outra execução em andamento (trava em `results/.pipeline.lock`) |

Alerta opcional em falha: exporte `ALERTA_WEBHOOK=https://...` e o wrapper faz um POST
com as últimas linhas do log.

## Logs e diagnóstico

```bash
tail -f logs/semanal-$(date +%F).log     # execução do dia
cat results/api/status.json              # saúde do pipeline
cat results/pipeline_state.json          # última semana processada
cat data/raw/download_report.json        # o que cada fonte devolveu
```

Logs com mais de 60 dias são apagados sozinhos. Se um job morreu no meio e deixou a trava,
ela é ignorada automaticamente depois de 6 horas — ou apague `results/.pipeline.lock`.

## Contrato para a API em Go

`results/api/` é reescrito por completo a cada execução (escrita atômica via `.tmp` +
rename, então a API nunca lê um arquivo pela metade).

### `previsao.json`

```json
{
  "modelo": "ARIMA",
  "criterio_selecao": "menor RMSE walk-forward em h=1",
  "rmse_walkforward": 0.0725,
  "ultima_semana_observada": "2026-08-16",
  "preco_observado_ultima_semana": 6.89,
  "semana_prevista": "2026-08-23",
  "horizonte": "1 semana",
  "previsao_pontual": 6.882,
  "p10": 6.83,
  "p90": 6.94,
  "probabilidades": { "p_alta": 0.09, "p_estavel": 0.61, "p_queda": 0.30 },
  "previsoes_por_modelo": { "naive": 6.89, "ARIMA": 6.882, "LightGBM": 6.859 },
  "ranking_h1": [ { "model": "ARIMA", "rmse": 0.0725, "mae": 0.0278 } ]
}
```

`p10`, `p90` e as probabilidades são `null` enquanto não houver pelo menos 20 resíduos
walk-forward do modelo vencedor. Todo `float` pode vir `null` — use ponteiros no Go.

### `historico.json`

`serie` traz a série semanal completa da ANP (`data`, `revenda`); `previsoes` traz uma
entrada por semana prevista, com `preco_realizado` e `erro` preenchidos assim que a ANP
publica aquela semana. É o suficiente para o front mostrar o gráfico e a acurácia real
acumulada.

### `status.json`

`status` é `atualizado`, `sem_novidade`, `dados_atualizados` ou `erro`. Use
`ultima_execucao_ok` e `ultima_semana_processada` para um health check: se
`ultima_semana_processada` ficar mais de ~10 dias atrás do dia de hoje, algo travou.

### Structs

```go
type Previsao struct {
    Modelo                     string             `json:"modelo"`
    RMSEWalkforward            float64            `json:"rmse_walkforward"`
    UltimaSemanaObservada      string             `json:"ultima_semana_observada"`
    PrecoObservadoUltimaSemana float64            `json:"preco_observado_ultima_semana"`
    SemanaPrevista             string             `json:"semana_prevista"`
    PrevisaoPontual            float64            `json:"previsao_pontual"`
    P10                        *float64           `json:"p10"`
    P90                        *float64           `json:"p90"`
    Probabilidades             map[string]*float64 `json:"probabilidades"`
    PrevisoesPorModelo         map[string]*float64 `json:"previsoes_por_modelo"`
}

type Status struct {
    Status                  string  `json:"status"`
    UltimaExecucaoOK        *string `json:"ultima_execucao_ok"`
    UltimaSemanaProcessada  *string `json:"ultima_semana_processada"`
    ModeloProducao          *string `json:"modelo_producao"`
    Erro                    *string `json:"erro"`
}
```

Os arquivos mudam no máximo uma vez por semana: sirva com cache e recarregue por
`mtime` em vez de reler a cada requisição.

## Subindo a API em Go

O serviço está em [`api/`](../api). Ele não tem dependência externa — só a stdlib —
e não interpreta o payload: entrega os bytes do JSON que o pipeline escreveu. Mudar
um campo no lado do Python não exige recompilar nada aqui.

### 1. Compilar

Na própria VM:

```bash
sudo apt install -y golang-go
cd /opt/TCC-modelo1/api
go build -o diesel-api .
```

Ou compile no Windows e mande só o binário (evita instalar Go na VM):

```powershell
cd api                                    # o go.mod fica aqui, nao na raiz
$env:GOOS="linux"; $env:GOARCH="amd64"; go build -o diesel-api .
scp diesel-api azureuser@<ip>:/opt/TCC-modelo1/api/
ssh azureuser@<ip> "chmod +x /opt/TCC-modelo1/api/diesel-api && sudo systemctl restart diesel-api"
```

O binario sai estatico (~8,8 MB), sem dependencia de libc — roda em qualquer Ubuntu.

### 2. systemd

```bash
sudo cp deploy/diesel-api.service /etc/systemd/system/
sudo nano /etc/systemd/system/diesel-api.service   # ajuste DIESEL_CORS_ORIGINS
sudo systemctl daemon-reload
sudo systemctl enable --now diesel-api
systemctl status diesel-api
curl -s localhost:8080/health
```

`DIESEL_CORS_ORIGINS` é a origem exata do front (`https://seuapp.vercel.app`), com
esquema e sem barra final. Várias, separadas por vírgula. `*` libera qualquer origem —
aceitável aqui, já que são dados públicos e só de leitura, mas prefira a lista.

### 3. DNS e firewall da Azure

Registro **A** para `api.seudominio.com.br` apontando para o IP público da VM.
E abra as portas no **Network Security Group** — o firewall da Azure é separado do
`ufw`, e esquecer disso é o motivo nº 1 de "o nginx subiu mas não responde":

```bash
az network nsg rule create -g <grupo> --nsg-name <nsg> -n allow-http \
  --priority 1001 --destination-port-ranges 80 443 --access Allow --protocol Tcp
```

Confirme que o DNS propagou antes de emitir o certificado: `dig +short api.seudominio.com.br`.

### 4. nginx e TLS

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
sudo cp deploy/nginx-api.conf /etc/nginx/sites-available/diesel-api
sudo sed -i 's/api.SEUDOMINIO.com.br/api.seudominio.com.br/g' /etc/nginx/sites-available/diesel-api
sudo ln -s /etc/nginx/sites-available/diesel-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d api.seudominio.com.br
```

O certbot escreve o bloco HTTPS e o redirect 80→443 sozinho, com os caminhos de
certificado corretos, e instala a renovação automática. Por isso o arquivo versionado
só tem a porta 80.

### 5. Conferir

```bash
curl -s https://api.seudominio.com.br/health
curl -s https://api.seudominio.com.br/api/previsao | head -20
curl -sI https://api.seudominio.com.br/api/previsao | grep -i etag
```

No front:

```js
const r = await fetch("https://api.seudominio.com.br/api/previsao");
const previsao = await r.json();
// previsao.modelo, previsao.previsao_pontual, previsao.p10, previsao.p90
```

### Comportamento que importa

- **`/health`** reprova por **atraso de dados**, não por erro pontual: uma falha de rede
  numa execução do cron não derruba a API enquanto a última previsão ainda vale.
  Retorna 503 quando `ultima_semana_processada` passa de `DIESEL_MAX_ATRASO` dias (10).
- **Cache por `mtime` + `ETag`**: o arquivo só é relido do disco quando o cron o reescreve.
  O front que mandar `If-None-Match` recebe `304` e não baixa nada.
- **Arquivo ainda não gerado** → `503` com mensagem explícita, não `500`.
- A API **nunca lê JSON pela metade**: o pipeline escreve com `.tmp` + rename atômico.
