# Street Rush AI

Sistema de visão computacional para analisar o **Street Rush** do Last War (cliente oficial para PC) e, no futuro, jogar de forma autônoma buscando a maior pontuação.

Esta versão **captura o monitor integral em segundo plano** e valida detecções visuais. Não envia teclado/mouse, não altera o cliente e não manipula memória ou rede.

## Requisitos

- Windows (cliente Last War)
- Python 3.10+
- Monitor do jogo (referência do projeto: **1918 × 1078** — a captura usa a resolução real do MSS)

## Comando definitivo (com salvamento verificável)

```bash
# 1) Teste controlado: 1 frame REAL do MSS → salva e encerra
python -m src.main --diag-capture

# 2) Loop contínuo (NÃO encerra sozinho — use Ctrl+C)
python -m src.main --save-every 30

# 3) Debug offline: boxes HSV nos frames reais → data/debug/
python -m src.main --render-debug

# 4) Debug ROI da pista (máscara + overlay + detecções core/soft/outside)
python -m src.main --render-roi-debug
```



Preview desligado por padrão. O processo continua ativo com o Last War em primeiro plano.
Sem `--save-every` / `--save-interval`, **nenhum PNG é gravado** (o loop só processa e loga).
Ao encerrar com Ctrl+C, o terminal mostra um resumo (frames, saves, FPS e caminhos absolutos).

**Importante:** os PNGs vão para o caminho absoluto de `CAPTURES_DIR` (raiz do código), que pode diferir do `data/captures` relativo ao cwd do terminal. O programa imprime `cwd`, `PROJECT_ROOT` e `CAPTURES_DIR` na abertura.

**Versionamento:** `data/captures/` e `data/debug/` (incluindo `roi/`) são commitados de propósito, para revisão visual no GitHub e por outras IAs. Não use `git add` seletivo que omita esses artefatos após gerar debug.

## Fluxo recomendado (Alt+Tab)

1. Abra o Cursor e o terminal integrado.
2. Execute: `python -m src.main --save-every 30`
3. Confira no terminal: monitor, resolução MSS e o **caminho absoluto** de `data/captures`.
4. Use **Alt+Tab** para colocar o **Last War** em primeiro plano.
5. O Python continua capturando/salvando em segundo plano (mensagem a cada PNG).
6. Volte ao terminal (Alt+Tab) e use **Ctrl+C** para ver o **resumo final**.

Não é necessário manter o Cursor em primeiro plano. Não há seletor de região (`selectROI`).

### Salvar frames com o jogo em foco

Como o teclado vai para o Last War, use um destes métodos:

```bash
# Ao iniciar: a cada 5 segundos
python -m src.main --save-interval 5

# Ao iniciar: a cada 60 frames
python -m src.main --save-every 60

# Em runtime (sem roubar foco): crie o arquivo-sentinela
# PowerShell:
New-Item -ItemType File -Force data\captures\.save_now
```

Frames são gravados em `data/captures/`.

### Preview opcional (calibração)

```bash
python -m src.main --preview
```

Abre uma janela pequena no canto, sem alterar a captura integral. Preferível em segundo monitor ou só para calibração — pode cobrir parte do jogo se estiver no mesmo monitor.

### Outras opções

```bash
python -m src.main --stats-interval 1
python -m src.main --dump-state
python -m src.main --monitor 1
python -m src.main --image data/captures/algum_frame.png
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
│   ├── screen_capture.py    # Captura integral do monitor (MSS)
│   ├── vision.py            # Reconhecimento visual
│   ├── game_state.py        # Estado do jogo
│   └── config.py            # Configurações centralizadas
├── tests/
├── data/captures/
├── models/
├── requirements.txt
└── README.md
```

## Monitoramento no terminal

A cada `--stats-interval` (padrão 1 s) o log mostra:

- Índice e resolução do monitor (MSS)
- Resolução do frame capturado
- FPS de captura
- Frames processados
- Tempo médio de processamento (ms)
- Contagem de erros de captura
- Resumo de objetos detectados

## Compatibilidade MSS × Last War

| Modo do jogo | Expectativa com MSS | Ação recomendada |
|--------------|---------------------|------------------|
| Janela | Em geral funciona | OK |
| Maximizado | Em geral funciona | OK |
| Tela cheia *borderless* / janela sem bordas | Em geral funciona (desktop composition) | **Preferido** |
| Tela cheia *exclusiva* (DirectX exclusive fullscreen) | Pode capturar tela preta, frame antigo ou falhar | Evitar; usar borderless / janela |

### Limitação conhecida

MSS lê o framebuffer do desktop (BitBlt / Desktop Duplication conforme backend). Em **fullscreen exclusivo**, o jogo pode apresentar direto na GPU sem passar pelo desktop — a captura pode não ver o frame real.

**Solução compatível (sem modificar o cliente):** nas opções gráficas do Last War / Windows, use modo **janela sem bordas (borderless)** ou **janela maximizada** na resolução nativa do monitor. Não alteramos arquivos do cliente nem memória do processo.

Esta limitação **não** é coberta pelos testes automatizados da CI/Linux; valide no Windows com o jogo aberto (salve um frame e confira se a imagem mostra o Street Rush).

## O que os testes automatizados cobrem

```bash
pytest
```

| Cobertura automatizada | Fora do escopo automatizado (manual) |
|------------------------|--------------------------------------|
| Geometria dinâmica do MSS (mock) | Alt+Tab com Last War real |
| Captura integral sem ROI | Fullscreen exclusivo vs borderless |
| Loop sem preview | Qualidade visual dos frames do jogo |
| Stats / erros / save sentinela | Calibração HSV com arte do Street Rush |
| Análise offline de imagem | Desempenho em 1918×1078 no hardware do usuário |

## Visão e estado

- `vision.py`: personagem, caminhões, barreiras, moedas (HSV + contornos). Só objetos realmente detectados.
- `game_state.py`: distância (fluxo óptico quando houver evidência), posição do personagem, obstáculos relativos, moedas.

Calibre HSV em `src/config.py` com frames reais salvos de `data/captures/`.

## Restrições

- Sem teclado/mouse no jogo
- Sem modificação do cliente oficial
- Sem leitura de memória / pacotes de rede

## Licença / uso

Projeto experimental para pesquisa de visão computacional. Use por sua conta e risco, respeitando os termos do Last War.
