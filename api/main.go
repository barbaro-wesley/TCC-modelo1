// API de leitura dos resultados do pipeline semanal do Diesel B S-10.
//
// O job em Python reescreve results/api/*.json no maximo uma vez por semana,
// sempre com .tmp + rename atomico. Esta API so precisa entregar esses bytes:
// ela nao interpreta o payload (exceto o status.json, no /health), entao mudar
// um campo do lado do Python nao exige recompilar nada aqui.
//
// Configuracao por ambiente:
//
//	DIESEL_DATA_DIR       diretorio dos JSON (default: ../results/api)
//	DIESEL_ADDR           endereco de escuta (default: 127.0.0.1:8080)
//	DIESEL_CORS_ORIGINS   origens permitidas, separadas por virgula, ou * (default: *)
//	DIESEL_MAX_ATRASO     dias sem semana nova antes do /health reprovar (default: 10)
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	arquivoPrevisao  = "previsao.json"
	arquivoHistorico = "historico.json"
	arquivoStatus    = "status.json"
)

// cacheArquivo serve o conteudo de um JSON do disco, relendo apenas quando o
// mtime muda. Os arquivos so mudam quando o cron retreina, entao reler a cada
// request seria desperdicio puro.
type cacheArquivo struct {
	nome string

	mu         sync.RWMutex
	dados      []byte
	etag       string
	modificado time.Time
}

func (c *cacheArquivo) carregar(dir string) ([]byte, string, error) {
	caminho := filepath.Join(dir, c.nome)
	info, err := os.Stat(caminho)
	if err != nil {
		return nil, "", err
	}

	c.mu.RLock()
	if c.dados != nil && info.ModTime().Equal(c.modificado) {
		dados, etag := c.dados, c.etag
		c.mu.RUnlock()
		return dados, etag, nil
	}
	c.mu.RUnlock()

	c.mu.Lock()
	defer c.mu.Unlock()
	// Outra goroutine pode ter recarregado enquanto trocavamos de lock.
	if c.dados != nil && info.ModTime().Equal(c.modificado) {
		return c.dados, c.etag, nil
	}
	dados, err := os.ReadFile(caminho)
	if err != nil {
		return nil, "", err
	}
	c.dados = dados
	c.modificado = info.ModTime()
	c.etag = fmt.Sprintf("%q", fmt.Sprintf("%d-%d", info.ModTime().UnixNano(), len(dados)))
	return c.dados, c.etag, nil
}

type servidor struct {
	dir        string
	origens    []string
	maxAtraso  int
	previsao   *cacheArquivo
	historico  *cacheArquivo
	statusFile *cacheArquivo
}

// statusPipeline e o subconjunto do status.json que o /health precisa entender.
type statusPipeline struct {
	Status                 string  `json:"status"`
	UltimaSemanaProcessada *string `json:"ultima_semana_processada"`
	UltimaExecucaoOK       *string `json:"ultima_execucao_ok"`
	ModeloProducao         *string `json:"modelo_producao"`
	Erro                   *string `json:"erro"`
}

func escreverJSON(w http.ResponseWriter, codigo int, corpo any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(codigo)
	if err := json.NewEncoder(w).Encode(corpo); err != nil {
		log.Printf("falha ao escrever resposta: %v", err)
	}
}

func erroJSON(w http.ResponseWriter, codigo int, msg string) {
	escreverJSON(w, codigo, map[string]string{"erro": msg})
}

func (s *servidor) servirArquivo(c *cacheArquivo) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD, OPTIONS")
			erroJSON(w, http.StatusMethodNotAllowed, "use GET")
			return
		}
		dados, etag, err := c.carregar(s.dir)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				erroJSON(w, http.StatusServiceUnavailable,
					"o pipeline ainda nao gerou "+c.nome)
				return
			}
			log.Printf("erro lendo %s: %v", c.nome, err)
			erroJSON(w, http.StatusInternalServerError, "falha ao ler o resultado")
			return
		}

		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("ETag", etag)
		// 5 min: os dados mudam 1x por semana, mas um retreino manual deve
		// aparecer rapido no front.
		w.Header().Set("Cache-Control", "public, max-age=300")
		if match := r.Header.Get("If-None-Match"); match != "" && match == etag {
			w.WriteHeader(http.StatusNotModified)
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(dados)))
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusOK)
			return
		}
		if _, err := w.Write(dados); err != nil {
			log.Printf("falha ao enviar %s: %v", c.nome, err)
		}
	}
}

// health reprova por atraso de dados, nao por erro pontual: uma falha de rede
// numa execucao nao significa que a API deva sair do ar enquanto a ultima
// previsao ainda for valida.
func (s *servidor) health(w http.ResponseWriter, r *http.Request) {
	dados, _, err := s.statusFile.carregar(s.dir)
	if err != nil {
		escreverJSON(w, http.StatusServiceUnavailable, map[string]any{
			"ok":     false,
			"motivo": "status.json indisponivel: " + err.Error(),
		})
		return
	}
	var st statusPipeline
	if err := json.Unmarshal(dados, &st); err != nil {
		escreverJSON(w, http.StatusServiceUnavailable, map[string]any{
			"ok":     false,
			"motivo": "status.json ilegivel: " + err.Error(),
		})
		return
	}

	corpo := map[string]any{
		"ok":                       true,
		"pipeline_status":          st.Status,
		"ultima_semana_processada": st.UltimaSemanaProcessada,
		"ultima_execucao_ok":       st.UltimaExecucaoOK,
		"modelo_producao":          st.ModeloProducao,
	}
	if st.Erro != nil {
		corpo["ultimo_erro"] = *st.Erro
	}

	if st.UltimaSemanaProcessada == nil {
		corpo["ok"] = false
		corpo["motivo"] = "pipeline nunca processou uma semana"
		escreverJSON(w, http.StatusServiceUnavailable, corpo)
		return
	}
	semana, err := time.Parse("2006-01-02", *st.UltimaSemanaProcessada)
	if err != nil {
		corpo["ok"] = false
		corpo["motivo"] = "data invalida em ultima_semana_processada"
		escreverJSON(w, http.StatusServiceUnavailable, corpo)
		return
	}
	atraso := int(time.Since(semana).Hours() / 24)
	corpo["dias_desde_ultima_semana"] = atraso
	if atraso > s.maxAtraso {
		corpo["ok"] = false
		corpo["motivo"] = fmt.Sprintf("dados parados ha %d dias (limite %d)", atraso, s.maxAtraso)
		escreverJSON(w, http.StatusServiceUnavailable, corpo)
		return
	}
	escreverJSON(w, http.StatusOK, corpo)
}

func (s *servidor) indice(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		erroJSON(w, http.StatusNotFound, "rota inexistente")
		return
	}
	escreverJSON(w, http.StatusOK, map[string]any{
		"servico": "Previsao semanal do Diesel B S-10 (ANP)",
		"rotas": []string{
			"GET /api/previsao",
			"GET /api/historico",
			"GET /api/status",
			"GET /health",
		},
		"fonte": "ANP SHLP semanal; previsao gerada pelo pipeline em Python",
	})
}

func (s *servidor) origemPermitida(origem string) string {
	if len(s.origens) == 1 && s.origens[0] == "*" {
		return "*"
	}
	for _, permitida := range s.origens {
		if permitida == origem {
			return origem
		}
	}
	return ""
}

func (s *servidor) cors(proximo http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if origem := r.Header.Get("Origin"); origem != "" {
			if permitida := s.origemPermitida(origem); permitida != "" {
				w.Header().Set("Access-Control-Allow-Origin", permitida)
			}
		}
		w.Header().Add("Vary", "Origin")
		if r.Method == http.MethodOptions {
			w.Header().Set("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, If-None-Match")
			w.Header().Set("Access-Control-Max-Age", "86400")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		proximo.ServeHTTP(w, r)
	})
}

type respostaComStatus struct {
	http.ResponseWriter
	codigo int
}

func (r *respostaComStatus) WriteHeader(codigo int) {
	r.codigo = codigo
	r.ResponseWriter.WriteHeader(codigo)
}

func registrar(proximo http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		inicio := time.Now()
		rw := &respostaComStatus{ResponseWriter: w, codigo: http.StatusOK}
		proximo.ServeHTTP(rw, r)
		log.Printf("%s %s %d %s", r.Method, r.URL.Path, rw.codigo, time.Since(inicio).Round(time.Millisecond))
	})
}

func env(chave, padrao string) string {
	if v := strings.TrimSpace(os.Getenv(chave)); v != "" {
		return v
	}
	return padrao
}

func dirPadrao() string {
	// Sem DIESEL_DATA_DIR, assume o layout do repo: api/ e results/ sao irmaos.
	exe, err := os.Executable()
	if err == nil {
		candidato := filepath.Join(filepath.Dir(exe), "..", "results", "api")
		if info, err := os.Stat(candidato); err == nil && info.IsDir() {
			return filepath.Clean(candidato)
		}
	}
	return filepath.Join("..", "results", "api")
}

func main() {
	log.SetFlags(log.LstdFlags | log.LUTC)

	dir := env("DIESEL_DATA_DIR", dirPadrao())
	addr := env("DIESEL_ADDR", "127.0.0.1:8080")
	origens := strings.Split(env("DIESEL_CORS_ORIGINS", "*"), ",")
	for i := range origens {
		origens[i] = strings.TrimSpace(origens[i])
	}
	maxAtraso, err := strconv.Atoi(env("DIESEL_MAX_ATRASO", "10"))
	if err != nil || maxAtraso <= 0 {
		log.Fatalf("DIESEL_MAX_ATRASO invalido: %q", os.Getenv("DIESEL_MAX_ATRASO"))
	}

	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		log.Fatalf("diretorio de dados inacessivel: %s (defina DIESEL_DATA_DIR)", dir)
	}

	s := &servidor{
		dir:        dir,
		origens:    origens,
		maxAtraso:  maxAtraso,
		previsao:   &cacheArquivo{nome: arquivoPrevisao},
		historico:  &cacheArquivo{nome: arquivoHistorico},
		statusFile: &cacheArquivo{nome: arquivoStatus},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.indice)
	mux.HandleFunc("/health", s.health)
	mux.HandleFunc("/api/previsao", s.servirArquivo(s.previsao))
	mux.HandleFunc("/api/historico", s.servirArquivo(s.historico))
	mux.HandleFunc("/api/status", s.servirArquivo(s.statusFile))

	srv := &http.Server{
		Addr:              addr,
		Handler:           registrar(s.cors(mux)),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	parar := make(chan os.Signal, 1)
	signal.Notify(parar, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("servindo %s em %s (CORS: %s)", dir, addr, strings.Join(origens, ", "))
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("servidor caiu: %v", err)
		}
	}()

	<-parar
	log.Println("encerrando...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("shutdown forcado: %v", err)
	}
	log.Println("encerrado")
}
