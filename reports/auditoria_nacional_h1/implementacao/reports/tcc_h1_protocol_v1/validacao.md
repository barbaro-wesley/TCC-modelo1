# Validação dos itens 1 e 2

Concluída em 12/09/2026, em cópia isolada do repositório, incluindo a alteração local preexistente da API. Commit base: `d807696953e2d4140b66b79aa76d12913e1e8840`; os hashes do manifesto identificam o código efetivamente testado, incluindo a implementação adicionada.

- Experimento: `9afa1b470bcb0b4a5a02ccd78589babe1e1003b2f1ddb1f881d4ae5dc7f96a86`.
- 118 testes e 19 subtestes passaram em 155,23 segundos.
- Ruff passou nos três arquivos Python novos.
- Smoke real: primeiro bloco, duas semanas por modelo, três modelos, 26 semanas de calibração anterior por modelo. Concluído sem erro; nenhum holdout foi avaliado ou modelo promovido.
- Os testes de perturbação preservaram as previsões anteriores à alteração de preços futuros; a observação corrente não afetou a previsão já emitida.
- O teste com fallback forçado reproduziu ponto, limites e estado posterior de uma execução direta da produção.
- O inicializador compartilhado reproduziu exatamente a previsão do caminho anterior usando ajuste real do statsmodels.
- Manifesto, snapshot, mudanças no código/ambiente, semanas ausentes e tentativa de sobrescrita foram verificados.

Comando da regressão (no repositório de teste):

```powershell
python -m pytest -q -p no:cacheprovider -W ignore::DeprecationWarning tests/test_frozen_experiment.py tests/test_s10_selection.py tests/test_pinned_windows.py tests/test_s10_production.py tests/test_training_corrections.py tests/test_promote_release.py tests/test_anp_official.py tests/test_calibration.py tests/test_online_learning.py tests/test_walkforward.py
```

Variáveis: `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

Saída final observada: `118 passed, 19 subtests passed in 155.23s (0:02:35)`. Este documento registra a execução observada; não é transcrição completa do log. A suíte inteira não foi executada. Os avisos de depreciação foram ocultados pelo comando.

O smoke está em `reports/tcc_h1_smoke_v1`. Seus seis erros servem para verificar funcionamento, não para escolher o melhor modelo. A avaliação completa dos 156 alvos por modelo é a próxima etapa.
