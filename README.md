# Street Rush AI

Sistema de visão computacional para analisar o **Street Rush** do Last War (cliente oficial para PC) e, no futuro, jogar de forma autônoma buscando a maior pontuação.

Esta primeira versão **apenas captura a tela e valida detecções visuais**. Não envia teclado/mouse, não altera o cliente e não manipula memória ou rede.

## Requisitos

- Windows (cliente Last War)
- Python 3.10+
- Resolução alvo do jogo: **1918 × 1078**

## Estrutura

```
street-rush-ai/
├── src/
│   ├── main.py              # Ponto de entrada
│   ├── screen_capture.py    # Captura MSS + preview/FPS
│   ├── vision.py            # Detecção e bounding boxes
│   ├── game_state.py        # Estado estruturado do jogo
│   └── config.py            # Configurações centralizadas
├── tests/
├── data/                    # Capturas e amostras
├── models/                  # Modelos futuros
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalação

```bash
cd street-rush-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

### Captura contínua com preview

```bash
python -m src.main
```

Teclas no preview:

| Tecla | Ação |
|-------|------|
| `q` | Sair |
| `r` | Selecionar região do jogo (ROI) |
| `s` | Salvar frame em `data/captures/` |

### Região fixa (sem tela inteira)

```bash
python -m src.main --region 0,0,1918,1078
```

### Seleção interativa de ROI na abertura

```bash
python -m src.main --select-region
```

### Analisar imagem estática (calibração / CI)

```bash
python -m src.main --image data/captures/frame_000001.png --no-preview
```

### Dump do estado (JSON)

```bash
python -m src.main --dump-state
```

## Testes

```bash
pytest
```

## Módulos

### Captura (`screen_capture.py`)

- OpenCV + NumPy + MSS
- Captura contínua
- Janela de preview independente com FPS
- Região configurável (`--region` ou tecla `r`)

### Visão (`vision.py`)

Detecta (quando houver evidência no frame):

- Personagem
- Caminhões
- Barreiras / obstáculos
- Moedas

Delimita com bounding boxes e calcula posição relativa ao personagem, aproximando a perspectiva 3D pela posição vertical na imagem.

**Importante:** objetos só entram no resultado se forem detectados de fato. Não há coordenadas inventadas. Os intervalos HSV em `src/config.py` são pontos de partida — calibre com capturas reais do Street Rush.

### Estado (`game_state.py`)

Representação estruturada com:

- Distância percorrida (estimativa por fluxo óptico, ou `null` sem evidência)
- Posição do personagem
- Objetos detectados
- Posição relativa dos obstáculos
- Moedas no percurso

## Calibração HSV

1. Rode o jogo em 1918×1078 e selecione a ROI.
2. Salve frames com `s`.
3. Ajuste `character_hsv`, `truck_hsv`, `barrier_hsv` e `coin_hsv` em `src/config.py`.
4. Valide com `python -m src.main --image ...`.

## Restrições (v0.1)

- Sem controle de teclado/mouse
- Sem modificação do cliente oficial
- Sem leitura de memória / pacotes de rede

## Licença / uso

Projeto experimental para pesquisa de visão computacional. Use por sua conta e risco, respeitando os termos do Last War.
