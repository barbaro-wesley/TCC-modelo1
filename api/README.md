# Backend Diesel S10

Python 3.13+, FastAPI, SQLAlchemy 2, PostgreSQL/Neon, Alembic e Redis. A API lê
publicações completas no PostgreSQL; não retreina modelos ao atender requisições.
O backend não depende dos pacotes de Machine Learning em `requirements.txt` da raiz.

## Execução local (PowerShell, na raiz do repositório)

```powershell
python -m venv .venv-api
.venv-api/Scripts/python.exe -m pip install -r api/requirements-dev.txt
Copy-Item .env.example .env
# Edite .env com URLs e JWT secret gerado aleatoriamente.
docker compose up -d --wait
.venv-api/Scripts/python.exe -m alembic -c api/alembic.ini upgrade head
.venv-api/Scripts/python.exe -m api.app.cli create-admin --email admin@example.com
.venv-api/Scripts/python.exe -m uvicorn api.app.main:create_app --factory --host 127.0.0.1 --port 8080
```

Linux: use `.venv-api/bin/python`. O comando de administrador solicita senha sem
mostrá-la no terminal e nunca altera um usuário já existente. Não há cadastro público
nem credencial administrativa padrão. Swagger local: `http://127.0.0.1:8080/docs`.

## Neon e Redis

Configure `S10_DATABASE_URL` com a connection string **pooled** do Neon, preservando
`sslmode=require` e os parâmetros de segurança fornecidos pelo serviço. O backend
normaliza `postgresql://` para o driver psycopg. Configure a URL **direta** em
`S10_MIGRATION_DATABASE_URL` para executar Alembic. Em produção, TLS PostgreSQL é obrigatório.
Migrations são executadas explicitamente antes do serviço, nunca por cada worker.

`S10_REDIS_URL` aceita `redis://` local e `rediss://` remoto com TLS e senha.
Use uma instância privada, compartilhada por todos os workers, com política
`noeviction`; não exponha 5432/6379 à internet. Prefixos Redis diferentes separam
desenvolvimento/homologação/produção. Reiniciar ou perder Redis pode reiniciar o rate
limit técnico, mas nunca apaga o consumo comercial mantido no PostgreSQL.

O Compose é apenas para desenvolvimento e publica PostgreSQL/Redis no loopback.
Quando usar Neon, basta `docker compose up -d redis` ou um Redis gerenciado.
As URLs são segredos: ficam no `.env` ignorado pelo Git. Não as envie no chat nem em logs.

Referências: [Neon connection pooling](https://neon.com/docs/connect/connection-pooling),
[Redis rate limiter](https://redis.io/docs/latest/develop/use-cases/rate-limiter/) e
[FastAPI JWT/Argon2](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/).

## Fluxo para disponibilizar o piloto

1. Crie o administrador pelo CLI e faça `POST /api/v1/auth/login` com JSON `email` e `password`.
2. Use `Authorization: Bearer <access_token>` nas chamadas administrativas.
3. Crie plano em `POST /api/v1/admin/plans`: `name`, `price_cents` (inteiro em centavos BRL),
   `monthly_quota`, `rate_per_minute`, `max_users`, `api_access`.
4. Crie empresa em `POST /api/v1/admin/organizations` com `name`, `cnpj` e `owner`
   (`name`, `email`, `password`). O proprietário é criado na mesma transação.
5. Ative a assinatura em `PUT /api/v1/admin/organizations/{id}/subscription` com
   `plan_id`, `status`, `starts_at`, `ends_at` (ISO 8601 com fuso, por exemplo `+00:00`).
6. O proprietário faz login, cadastra usuários e consulta o produto.

Não há preço fictício ou plano pago criado automaticamente. Os preços são definidos
pelo administrador; o cadastro da assinatura é manual, sem gateway de pagamento,
webhooks ou emissão de fatura. Estados: `trialing`, `active`, `past_due`, `canceled`,
`suspended`. Apenas os dois primeiros, dentro do intervalo contratado, liberam o produto.
Plano é uma versão imutável: crie outra versão para mudar preço/limites; DELETE o
desativa no catálogo. A assinatura guarda snapshot, e um PUT explícito troca o contrato.
Uma troca de plano não zera o consumo do mês nem permite contratar menos assentos que
os usuários ativos. Datas do contrato e mês de consumo são conceitos separados.

## Identidade e autorização

Neste MVP, um usuário pertence a uma única empresa; administradores da plataforma não
pertencem a empresas. E-mail é único e normalizado. Papéis: `platform_admin`, `owner`,
`member`, `viewer`. API key identifica uma integração da empresa, não uma pessoa.

- Administrador: empresas, planos, assinaturas, consumo administrativo e auditoria.
- Proprietário: usuários e chaves da própria empresa, além de consultas ao produto.
- Membro: consultas e simulações; sem gestão de usuários/planos.
- Leitor: consultas; não executa POST de simulação.
- Integração: previsão, histórico, simulação, status, assinatura e consumo da empresa;
  nunca acessa administração ou gestão de usuários. Exige `api_access` no plano.

A empresa sempre vem da identidade autenticada. IDs de terceiros em rotas da empresa
resultam em 404. Contas desativadas não fazem login; tokens existentes param de funcionar.
Desativar empresa revoga sessões e chaves. Não há exclusão física de usuários/empresas
pela API, preservando auditoria; use PATCH `active:false`. O último proprietário ativo
não pode ser removido/rebaixado. Máximo de dez API keys ativas por empresa.

Senhas Argon2; access JWT expira em 15 minutos. Refresh opaco, armazenado por hash,
expira em 7 dias no total e gira a cada uso. Reutilizar refresh revogado invalida a
família inteira. Logout e redefinição de senha revogam sessões; cada access token
consulta a sessão e o usuário atuais. Tokens voltam em JSON, adequado a este backend;
ao construir o frontend será necessário definir armazenamento seguro/BFF ou cookies
HttpOnly com proteção CSRF. Nenhuma API key deve ser embutida no frontend.

Reset de senha do piloto: administrador autenticado emite token de uso único por
`POST /admin/users/{id}/password-reset`, entrega por canal seguro e o usuário chama
`POST /auth/reset-password` (`token`, `password`). Expira em 30 minutos. Não há envio
automático de e-mail nem endpoint público de recuperação/invites nesta versão.

## Limites e paginação

| Controle | Padrão | Configuração |
|---|---:|---|
| API inteira | 3.000 requisições/minuto | `S10_GLOBAL_RATE_PER_MINUTE` |
| IP | 120/minuto | `S10_IP_RATE_PER_MINUTE` |
| Autenticação por IP; login também por e-mail | 10/minuto | `S10_LOGIN_RATE_PER_MINUTE` |
| Empresa (todos os usuários e chaves juntos) | contratado | snapshot da assinatura |
| Tamanho de página padrão | 20 | `S10_PAGE_SIZE` |
| Tamanho máximo de página | 100 | `S10_MAX_PAGE_SIZE` |
| Offset máximo | 10.000 | `S10_MAX_OFFSET` |
| Tempo máximo por statement e espera de lock | 5 segundos | `S10_QUERY_TIMEOUT_MS` |
| Conexões PostgreSQL por worker | 5, sem overflow | `S10_POOL_SIZE` |
| Corpo HTTP máximo, inclusive chunked | 32 KiB | `S10_MAX_BODY_BYTES` |

Redis executa incremento+expiração atomicamente com Lua. Janela fixa de 60 segundos
a partir da primeira chamada; uma rajada pode ocorrer na transição entre janelas.
HTTP 429 retorna `Retry-After`, `RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset`
(segundos até reset), `X-RateLimit-Scope`. Os headers de sucesso autenticado mostram
o balde da empresa. Redis indisponível: 503, sem liberar acesso ilimitado.
Liveness e preflight CORS não consomem rate limit; outras rotas passam pelo limite global/IP.
Atrás de nginx, aceite forwarded headers apenas do proxy confiável; o template substitui
o X-Forwarded-For do cliente. Sem proxy, não habilite confiança global em headers.

Listagens: `?page=1&page_size=20`; resultado `items`, `total`, `page`, `page_size`, `pages`.
Ordenação estável por data e ID. Tamanho acima do limite e offset profundo retornam
422, sem clamp silencioso. `total`/`pages` são contagem do conjunto; páginas além do
offset máximo são inacessíveis. Não há `all=true` ou endpoint que devolva histórico
inteiro. Histórico aceita `series=observed|forecasts`, `start`, `end` (YYYY-MM-DD).

Cota comercial: uma unidade por resposta bem-sucedida em `GET /forecast`,
`GET /history` ou `POST /scenarios/cost`, tanto por usuário quanto por chave. Cada
página e cada nova chamada conta, incluindo repetições; sem idempotência comercial
nesta versão. Conta após processamento bem-sucedido, antes de enviar a resposta;
uma conexão interrompida após commit ainda pode ter consumido a unidade.
Status, perfil, administração e consulta de consumo não gastam cota.
Erros 4xx/5xx não geram unidade comercial, mas passam pelo rate limit técnico.

Contador e evento são confirmados na mesma transação PostgreSQL, com lock por empresa.
Cota esgotada: 429 `monthly_quota_exceeded`; assinatura inativa: 403
`subscription_inactive`. O período é mês civil UTC (`YYYY-MM`), com renovação por nova
linha, sem apagar histórico e sem cron de reset. Um Redis reiniciado não altera a cota.
Headers `X-Quota-Limit`, `X-Quota-Remaining`, `X-Quota-Period` acompanham sucesso.

## Rotas

Todas com prefixo `/api/v1`, exceto health e OpenAPI:

| Grupo | Rotas |
|---|---|
| Auth | POST `/auth/login`, `/auth/refresh`, `/auth/logout`, `/auth/reset-password`; GET `/me` |
| Empresas (admin) | GET/POST `/admin/organizations`; PATCH `/admin/organizations/{id}` |
| Planos | GET `/plans`; POST `/admin/plans`; DELETE `/admin/plans/{id}` |
| Assinaturas | PUT `/admin/organizations/{id}/subscription`; GET `/subscription` |
| Usuários | GET/POST `/organization/users`; PATCH `/organization/users/{id}` |
| Consulta administrativa | GET `/admin/users?organization_id=UUID`, `/admin/organizations/{id}`, `/admin/organizations/{id}/subscription` |
| Reset administrativo | POST `/admin/users/{id}/password-reset` |
| Chaves | GET/POST `/organization/api-keys`; DELETE `/organization/api-keys/{id}` |
| Consumo | GET `/usage`, `/usage/events`, `/admin/organizations/{id}/usage` |
| Auditoria | GET `/admin/audit-logs` |
| Produto | GET `/forecast`, `/history`, `/status`; POST `/scenarios/cost` |

`GET /health/live` verifica processo; `GET /health/ready` verifica banco migrado e
Redis. A saúde do modelo é independente: previsão ausente, inválida ou vencida
retorna 503 e não gasta cota. O histórico continua consultável. Os dados do modelo
existentes no repositório podem estar antigos; execute o pipeline para atualizá-los.
Respostas autenticadas usam `Cache-Control: no-store`, não há cache público nem
rotas antigas abertas. Produção não expõe Swagger; `/openapi.json` exige admin.
Logs HTTP não incluem corpo, query string, Authorization, senha ou chave.

## Testes

```powershell
docker compose up -d --wait
$env:S10_TEST_DATABASE_URL='postgresql://s10:s10-local-only@localhost:5432/s10'
$env:S10_TEST_REDIS_URL='redis://localhost:6379/1'
.venv-api/Scripts/python.exe -m pytest -c api/pytest.ini api/tests -q
.venv-api/Scripts/python.exe -m ruff check api
.venv-api/Scripts/python.exe -m ruff format --check api
```

Cada teste de integração cria e remove somente um schema aleatório `s10_test_*`,
aplica a migração real, usa conexões independentes e limpa apenas seu prefixo Redis.
Não use banco de produção para testes. Sem `S10_TEST_DATABASE_URL`, testes PostgreSQL
ficam explicitamente SKIPPED; testes de Redis usam fakeredis com Lua se não houver
`S10_TEST_REDIS_URL`. O workflow CI usa PostgreSQL e Redis reais.

## Publicação

Na VPS, configure `.env`, instale nginx/certbot e Python 3.13+, disponibilize Redis
privado e Neon e execute `sudo ./deploy/instalar_api.sh api.seudominio.com.br`.
O script mantém o backup nginx, instala dependências Python, aplica migrações e
atualiza o systemd existente. Adicione domínio e `127.0.0.1` aos hosts permitidos.
Use `chmod 600 .env`, propriedade do usuário do serviço. O processo escuta somente
127.0.0.1:8080; nginx termina HTTPS. Nenhum deploy remoto é executado automaticamente.

Alternativa: `docker build -f api/Dockerfile -t s10-api .`; forneça ambiente na
execução. Não precisa montar arquivos de resultados. O container roda sem
root. Migração é um comando separado. O rate limit total é compartilhado; o pool
do banco é por worker, então 2 workers com pool 5 usam até 10 conexões.

O job usa outro ambiente, credencial e imagem em `training/`. A API tem somente
SELECT nas tabelas `model_*`, além das permissões comerciais existentes.
Migre para `0002_model_publication` antes de iniciar esta versão. Forecast/history
respondem 503 até a primeira publicação; JSONs antigos não são importados.
Os endpoints preservam seus campos e acrescentam `run_id` à previsão e às linhas
de histórico de previsões. Filtros e paginação do histórico são feitos no SQL.
Procedimento de atualização, permissões e execução do job: [deploy/README.md](../deploy/README.md).
