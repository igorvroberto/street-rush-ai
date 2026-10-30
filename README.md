# Street Rush AI

Sistema de visão computacional para analisar o **Street Rush** do Last War (cliente oficial para PC) e, no futuro, jogar de forma autônoma buscando a maior pontuação.

Esta versão **captura o monitor integral em segundo plano** e valida detecções visuais. Não envia teclado/mouse, não altera o cliente e não manipula memória ou rede.

## Requisitos

- Windows (cliente Last War)
- Python 3.10+
- Monitor do jogo (referência do projeto: **1918 × 1078** — a captura usa a resolução real do MSS)

## Comandos de captura (sessões exclusivas)

Cada execução cria automaticamente uma pasta nova — **nunca sobrescreve** partidas anteriores:

```text
data/captures/session_YYYYMMDD_HHMMSS/
  frames/
    frame_000030.png
    frame_000060.png
    ...
  metadata.json

data/debug/session_YYYYMMDD_HHMMSS/
  hsv/
  roi/
  character/
```

```bash
# 1) Teste controlado: 1 frame REAL do MSS → nova sessão → encerra
python -m src.main --diag-capture

# 2) Loop contínuo (NÃO encerra sozinho — use Ctrl+C)
python -m src.main --save-every 30

# 3) Pasta de saída manual (ainda com frames/ + metadata.json)
python -m src.main --save-every 30 --output-dir data/captures/minha_partida_teste

# 4) Listar sessões gravadas
python -m src.main --list-sessions
```

Preview desligado por padrão. O processo continua ativo com o Last War em primeiro plano.
Sem `--save-every` / `--save-interval`, **nenhum PNG é gravado** (o loop só processa e loga).
Ao encerrar com Ctrl+C, o terminal mostra um resumo e atualiza `metadata.json` (data/hora, resolução, FPS, frames, parâmetros).

**Importante:** os PNGs vão para o caminho absoluto impresso no log (`frames → ...`). O programa também imprime `cwd`, `PROJECT_ROOT` e o id da sessão.

**Versionamento:** `data/captures/` e `data/debug/` são commitados de propósito, para revisão visual no GitHub e por outras IAs.

## Selecionar uma sessão para análise

```bash
# Listar
python -m src.main --list-sessions

# Debug HSV da sessão
python -m src.main --session session_20261030_183000 --render-debug

# Debug ROI
python -m src.main --session session_20261030_183000 --render-roi-debug

# Debug do detector de personagem
python -m src.main --session session_20261030_183000 --render-character-debug
```

Os resultados vão para `data/debug/session_.../{hsv,roi,character}/` — **não misturam** partidas.

### Compatibilidade com frames antigos

Frames soltos em `data/captures/*.png` (antes das sessões) continuam válidos. Sem `--session`, os comandos `--render-*-debug` usam esses PNGs legados e gravam em `data/debug/` (layout antigo).

## Fluxo recomendado (Alt+Tab)

1. Abra o Cursor e o terminal integrado.
2. Execute: `python -m src.main --save-every 30`
3. Confira no terminal: id da sessão e o caminho absoluto de `frames/`.
4. Use **Alt+Tab** para colocar o **Last War** em primeiro plano.
5. O Python continua capturando/salvando em segundo plano.
6. Volte ao terminal (Alt+Tab) e use **Ctrl+C** para o **resumo** + `metadata.json`.

Não é necessário manter o Cursor em primeiro plano. Não há seletor de região (`selectROI`).

### Salvar frames com o jogo em foco

```bash
# Ao iniciar: a cada 5 segundos
python -m src.main --save-interval 5

# Ao iniciar: a cada 60 frames
python -m src.main --save-every 60

# Em runtime (sem roubar foco): arquivo-sentinela na raiz de captures
# PowerShell:
New-Item -ItemType File -Force data\captures\.save_now
```

### Preview opcional (calibração)

```bash
python -m src.main --preview
```

### Outras opções

```bash
python -m src.main --stats-interval 1
python -m src.main --dump-state
python -m src.main --monitor 1
python -m src.main --image data/captures/session_.../frames/frame_000001.png
```

## Instalação

```bash
cd street-rush-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Estrutura

```
street-rush-ai/
├── src/
│   ├── main.py              # Coordenação / loop / monitoramento
│   ├── session.py           # Sessões exclusivas + metadata
│   ├── screen_capture.py    # Captura integral do monitor (MSS)
│   ├── vision.py            # Reconhecimento visual (HSV legado)
│   ├── character.py         # Detector de personagem (ROI)
│   ├── roi.py               # ROI da pista
│   ├── game_state.py        # Estado do jogo
│   └── config.py            # Configurações centralizadas
├── tests/
├── data/
│   ├── captures/
│   │   ├── frame_*.png              # legado (pré-sessão)
│   │   └── session_YYYYMMDD_HHMMSS/
│   │       ├── frames/
│   │       └── metadata.json
│   └── debug/
│       └── session_YYYYMMDD_HHMMSS/
│           ├── hsv/
│           ├── roi/
│           └── character/
├── models/
├── requirements.txt
└── README.md
```

## Comparação futura entre sessões

O `metadata.json` reserva campos em `comparison` (hit rate, confiança média, FP/FN, situações de corrida/salto/agachamento/faixa) para análise posterior. **Ainda não há comparação automática** — só a organização dos dados.

## Monitoramento no terminal

A cada `--stats-interval` (padrão 1 s) o log mostra índice/resolução do monitor, FPS, frames processados, erros e resumo de detecções.

## Compatibilidade MSS × Last War

| Modo do jogo | Expectativa com MSS | Ação recomendada |
|--------------|---------------------|------------------|
| Janela | Em geral funciona | OK |
| Maximizado | Em geral funciona | OK |
| Tela cheia *borderless* | Em geral funciona | **Preferido** |
| Tela cheia *exclusiva* | Pode falhar / tela preta | Evitar; usar borderless |

## O que os testes automatizados cobrem

```bash
pytest
```

Inclui: duas execuções criam sessões distintas sem sobrescrita; `metadata.json`; frames legados; captura mockada MSS; ROI; detector de personagem.

## Visão e estado

- `vision.py`: personagem HSV legado, caminhões, barreiras, moedas.
- `character.py`: detector de personagem na ROI (independente do HSV legado).
- `game_state.py`: distância (fluxo óptico), posição, obstáculos relativos.

## Restrições

- Sem teclado/mouse no jogo
- Sem modificação do cliente oficial
- Sem leitura de memória / pacotes de rede

## Licença / uso

Projeto experimental para pesquisa de visão computacional. Use por sua conta e risco, respeitando os termos do Last War.
