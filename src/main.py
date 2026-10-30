"""Ponto de entrada do Street Rush AI.

Fluxo recomendado (Windows):
1. Diagnóstico de 1 frame real: ``python -m src.main --diag-capture``
2. Loop contínuo com salvamento: ``python -m src.main --save-every 30``
3. Alt+Tab para o Last War; Ctrl+C no terminal para o resumo.

O loop **não encerra sozinho** — isso é esperado. Sem ``--save-every`` /
``--save-interval``, nenhum PNG é gravado.

Restrições: sem teclado/mouse no jogo; sem alterar o cliente; sem memória/rede.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CAPTURES_DIR,
    DEBUG_DIR,
    MONITOR_INDEX,
    PROJECT_ROOT,
    SAVE_EVERY_N_FRAMES,
    SAVE_INTERVAL_SEC,
    SAVE_REQUEST_FILE,
    STARTUP_DELAY_SEC,
    STATS_LOG_INTERVAL_SEC,
)
from src.game_state import GameStateTracker  # noqa: E402
from src.screen_capture import (  # noqa: E402
    KEY_QUIT,
    KEY_SAVE_FRAME,
    ScreenCapture,
    ScreenCaptureError,
)
from src.session import (  # noqa: E402
    SessionMetadata,
    SessionPaths,
    create_capture_session,
    create_session_from_output_dir,
    finalize_metadata,
    list_session_dirs,
    next_frame_path,
    resolve_session,
)
from src.vision import VisionAnalyzer, summarize_detections  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("street_rush_ai")


@dataclass
class RunSession:
    """Métricas acumuladas de uma execução (para resumo no Ctrl+C)."""

    captures_dir: Path
    frames_captured: int = 0
    frames_processed: int = 0
    images_saved: int = 0
    capture_errors: int = 0
    save_errors: int = 0
    saved_paths: List[Path] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    last_fps: float = 0.0

    @property
    def avg_fps(self) -> float:
        elapsed = time.perf_counter() - self.started_at
        if elapsed <= 0:
            return 0.0
        if self.last_fps > 0:
            return self.last_fps
        return self.frames_captured / elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Street Rush AI — captura integral do monitor + visão "
            "(loop contínuo até Ctrl+C; preview off por padrão)."
        )
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=MONITOR_INDEX,
        help="Índice do monitor MSS (padrão: 1 = principal).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Abre janela de preview pequena/deslocada (opcional; off por padrão).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Para após N frames (0 = infinito; o padrão é rodar até Ctrl+C).",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=STATS_LOG_INTERVAL_SEC,
        help="Intervalo em segundos para log de monitoramento no terminal.",
    )
    parser.add_argument(
        "--save-interval",
        type=float,
        default=SAVE_INTERVAL_SEC,
        help="Salva um frame a cada N segundos (0 = desligado).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY_N_FRAMES,
        help=(
            "Salva um frame a cada N frames processados (0 = desligado). "
            "Ex.: --save-every 30"
        ),
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=STARTUP_DELAY_SEC,
        help=(
            "Segundos de espera antes do 1º frame em novas capturas "
            f"(padrão: {STARTUP_DELAY_SEC:g}; 0 = sem espera). "
            "Não se aplica a --render-*-debug / --image / --list-sessions."
        ),
    )
    parser.add_argument(
        "--dump-state",
        action="store_true",
        help="Imprime o estado JSON junto com o log de stats.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Analisa uma imagem estática (validação offline / CI).",
    )
    parser.add_argument(
        "--diag-capture",
        action="store_true",
        help=(
            "Teste controlado: captura 1 frame REAL via MSS, salva PNG, "
            "imprime diagnóstico completo e encerra (sem loop infinito)."
        ),
    )
    parser.add_argument(
        "--render-debug",
        action="store_true",
        help=(
            "Gera imagens de debug (boxes + contagens) a partir dos frames "
            "reais em data/captures/ e salva em data/debug/ (sem captura ao vivo)."
        ),
    )
    parser.add_argument(
        "--render-roi-debug",
        action="store_true",
        help=(
            "Gera debug da ROI da pista (máscara, overlay, detecções core/soft/outside) "
            "em data/debug/roi/ a partir dos frames reais (sem captura ao vivo)."
        ),
    )
    parser.add_argument(
        "--render-character-debug",
        action="store_true",
        help=(
            "Gera debug do detector de personagem (REAL vs DETECTADO vs HSV legado) "
            "a partir dos frames reais (use --session para escolher a partida)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Pasta manual da sessão de captura (cria frames/ + metadata.json). "
            "Sem este flag, cria data/captures/session_YYYYMMDD_HHMMSS/."
        ),
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help=(
            "Sessão para análise offline (id session_YYYYMMDD_HHMMSS ou caminho). "
            "Usado com --render-*-debug."
        ),
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Lista sessões em data/captures/ e encerra.",
    )
    parser.add_argument(
        "--trace-frames",
        action="store_true",
        default=True,
        help="Loga cada frame recebido (padrão: ligado).",
    )
    parser.add_argument(
        "--no-trace-frames",
        action="store_false",
        dest="trace_frames",
        help="Desliga o log por frame (mantém só stats/saves).",
    )
    return parser


def announce(msg: str) -> None:
    """Mensagem visível no terminal (stdout), além do logger."""
    print(f"[street-rush-ai] {msg}", flush=True)
    logger.info("%s", msg)


def announce_path_diagnostics(captures_dir: Path) -> None:
    """Compara CWD do terminal com a raiz do projeto e o diretório de saída."""
    cwd = Path.cwd().resolve()
    project = PROJECT_ROOT.resolve()
    out = captures_dir.resolve()
    announce(f"cwd (terminal)          : {cwd}")
    announce(f"PROJECT_ROOT (código)   : {project}")
    announce(f"CAPTURES_DIR (saída)    : {out}")
    announce(f"cwd == PROJECT_ROOT?    : {cwd == project}")
    announce(f"saída dentro do projeto?: {project in out.parents or out.parent == project}")
    if cwd != project:
        announce(
            "AVISO: o terminal NÃO está na raiz do projeto. "
            "Os PNGs vão para CAPTURES_DIR (absoluto acima), "
            f"NÃO para {cwd / 'data' / 'captures'}."
        )
    relative_guess = cwd / "data" / "captures"
    if relative_guess.resolve() != out:
        announce(
            f"AVISO: '{relative_guess}' (relativo ao cwd) ≠ saída real '{out}'. "
            "Abra o caminho ABSOLUTO da saída no Explorer."
        )


def save_frame(
    frame: np.ndarray,
    frames: int,
    captures_dir: Path,
    *,
    reason: str = "save",
) -> Path:
    """Persiste um frame BGR com log explícito do resultado de ``cv2.imwrite``.

    Numeração sequencial ``frame_000001.png``; se o nome existir, usa sufixo
    incremental (nunca sobrescreve). Resolução original preservada.
    """
    captures_dir = captures_dir.resolve()
    captures_dir.mkdir(parents=True, exist_ok=True)
    out = next_frame_path(captures_dir, frames).resolve()

    announce(
        f"TENTATIVA DE SALVAMENTO | motivo={reason} | frame=#{frames} | "
        f"shape={frame.shape[1]}x{frame.shape[0]} | destino={out}"
    )

    imwrite_ok = bool(cv2.imwrite(str(out), frame))
    announce(f"cv2.imwrite retornou: {imwrite_ok} | path={out}")

    if not imwrite_ok or not out.is_file() or out.stat().st_size <= 0:
        announce("cv2.imwrite falhou ou arquivo vazio — tentando imencode+write_bytes")
        ok, buf = cv2.imencode(".png", frame)
        announce(f"cv2.imencode retornou: {ok}")
        if not ok:
            raise OSError(f"cv2.imwrite e cv2.imencode falharam para {out}")
        out.write_bytes(buf.tobytes())

    if not out.is_file() or out.stat().st_size <= 0:
        raise OSError(f"Arquivo não foi gravado corretamente: {out}")

    announce(
        f"RESULTADO OK | frame=#{frames} | bytes={out.stat().st_size} | "
        f"caminho_absoluto={out}"
    )
    return out


def capture_run_parameters(args: argparse.Namespace) -> dict:
    """Parâmetros da execução para ``metadata.json``."""
    return {
        "monitor": int(args.monitor),
        "save_every": int(args.save_every),
        "save_interval": float(args.save_interval),
        "max_frames": int(args.max_frames),
        "preview": bool(args.preview),
        "stats_interval": float(args.stats_interval),
        "dump_state": bool(args.dump_state),
        "output_dir": getattr(args, "output_dir", None),
        "startup_delay_sec": float(getattr(args, "startup_delay", STARTUP_DELAY_SEC)),
    }


def wait_startup_delay(seconds: float) -> float:
    """Contagem regressiva antes da 1ª captura; não chama MSS/visão.

    Retorna o tempo de espera efetivamente aplicado (>= 0).
    """
    delay = float(seconds)
    if delay <= 0:
        announce("startup-delay=0 — iniciando captura imediatamente.")
        return 0.0
    # Contagem inteira visível: 5, 4, 3, 2, 1 (arredonda para cima se fracionário).
    steps = max(1, int(np.ceil(delay)))
    announce(
        f"Aguardando {delay:g}s antes da captura (Alt+Tab para o Last War)…"
    )
    for remaining in range(steps, 0, -1):
        announce(f"  {remaining}")
        # Distribui o atraso total nos ticks da contagem.
        time.sleep(delay / float(steps))
    announce("Espera concluída — iniciando captura.")
    return delay


def open_capture_session(
    args: argparse.Namespace,
    *,
    captures_parent: Optional[Path] = None,
) -> tuple[SessionPaths, SessionMetadata]:
    """Abre sessão exclusiva (nova) ou pasta ``--output-dir`` manual."""
    params = capture_run_parameters(args)
    output = getattr(args, "output_dir", None)
    parent = (captures_parent or CAPTURES_DIR).resolve()
    # Em produção: data/debug/. Em testes com parent temporário: parent/debug/.
    debug_root = (
        DEBUG_DIR.resolve()
        if parent == CAPTURES_DIR.resolve()
        else (parent / "debug")
    )
    if output:
        return create_session_from_output_dir(
            Path(output),
            debug_root=debug_root,
            parameters=params,
        )
    return create_capture_session(
        captures_root=parent,
        debug_root=debug_root,
        parameters=params,
    )


def print_sessions_list(captures_root: Path = CAPTURES_DIR) -> int:
    sessions = list_session_dirs(captures_root)
    if not sessions:
        announce(f"Nenhuma sessão em {captures_root.resolve()}")
        return 0
    announce(f"Sessões em {captures_root.resolve()}:")
    for path in sessions:
        meta_path = path / "metadata.json"
        extra = ""
        if meta_path.is_file():
            try:
                meta = SessionMetadata.load(meta_path)
                extra = (
                    f" | frames_saved={meta.frames_saved} | "
                    f"res={meta.resolution} | fps={meta.fps_avg}"
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                extra = " | metadata ilegível"
        n_frames = len(list((path / "frames").glob("*.png"))) if (path / "frames").is_dir() else 0
        announce(f"  - {path.name} | pngs={n_frames}{extra}")
    announce("Análise: python -m src.main --session <id> --render-character-debug")
    return 0


def resolve_analysis_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Optional[SessionPaths]]:
    """Resolve pasta de frames e pasta de debug para comandos offline."""
    if getattr(args, "session", None):
        paths = resolve_session(args.session, captures_root=CAPTURES_DIR, debug_root=DEBUG_DIR)
        paths.debug_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("hsv", "roi", "character"):
            (paths.debug_dir / sub).mkdir(parents=True, exist_ok=True)
        return paths.frames_dir, paths.debug_dir, paths
    # Legado: frames soltos em data/captures/
    return CAPTURES_DIR, DEBUG_DIR, None


def should_save_frame(
    *,
    frames: int,
    save_every: int,
    save_interval: float,
    last_save_ts: float,
    now: float,
    sentinel: bool,
) -> Tuple[bool, str]:
    """Decide se o frame atual deve ser persistido e explica o motivo."""
    if sentinel:
        return True, "sentinela .save_now"
    if save_every > 0 and frames > 0 and frames % save_every == 0:
        return True, f"save_every (frame {frames} % {save_every} == 0)"
    if save_interval > 0 and (now - last_save_ts) >= save_interval:
        return True, f"save_interval ({save_interval}s)"
    return False, (
        f"nenhuma regra (frame={frames}, save_every={save_every}, "
        f"save_interval={save_interval})"
    )


def consume_save_request(sentinel_path: Path = SAVE_REQUEST_FILE) -> bool:
    """Retorna True se o arquivo-sentinela existir (e o remove)."""
    try:
        if sentinel_path.exists():
            sentinel_path.unlink(missing_ok=True)
            return True
    except OSError as exc:
        announce(f"ERRO ao ler/remover sentinela de save ({sentinel_path}): {exc}")
    return False


def log_stats(
    capture: ScreenCapture,
    detections_summary: str,
    session: RunSession,
    *,
    dump_state: bool = False,
    state_dict: dict | None = None,
) -> None:
    mon = capture.monitor
    stats = capture.stats
    w, h = stats.last_resolution
    announce(
        f"stats | monitor={mon.index} res={mon.resolution_label} "
        f"captura={w}x{h} fps={stats.fps:.1f} "
        f"capturados={session.frames_captured} processados={session.frames_processed} "
        f"salvos={session.images_saved} proc_ms={stats.avg_process_ms:.1f} "
        f"erros_captura={session.capture_errors} | {detections_summary}"
    )
    if dump_state and state_dict is not None:
        print(json.dumps(state_dict, ensure_ascii=False), flush=True)


def print_run_summary(session: RunSession, *, reason: str) -> None:
    """Resumo final (Ctrl+C, fim normal ou erro)."""
    lines = [
        "========== RESUMO DA EXECUÇÃO ==========",
        f"Motivo do encerramento : {reason}",
        f"Frames capturados      : {session.frames_captured}",
        f"Frames processados     : {session.frames_processed}",
        f"Imagens salvas         : {session.images_saved}",
        f"Erros de captura       : {session.capture_errors}",
        f"Erros de gravação      : {session.save_errors}",
        f"FPS médio              : {session.avg_fps:.1f}",
        f"Diretório de saída     : {session.captures_dir.resolve()}",
        f"cwd no encerramento    : {Path.cwd().resolve()}",
    ]
    if session.saved_paths:
        lines.append("Arquivos salvos:")
        for path in session.saved_paths:
            lines.append(f"  - {path}")
    else:
        lines.append(
            "Arquivos salvos: (nenhum — use --save-every N, --save-interval S, "
            "--diag-capture ou o sentinela .save_now)"
        )
    lines.append("========================================")
    text = "\n".join(lines)
    print(text, flush=True)
    logger.info("Resumo:\n%s", text)


def analyze_image(path: Path, show_preview: bool) -> int:
    """Modo offline: analisa um arquivo de imagem."""
    frame = cv2.imread(str(path))
    if frame is None:
        announce(f"ERRO: não foi possível ler a imagem: {path.resolve()}")
        return 1

    vision = VisionAnalyzer()
    tracker = GameStateTracker()
    t0 = time.perf_counter()
    result = vision.analyze(frame)
    state = tracker.update(frame, result, fps=0.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    announce(
        f"Imagem {path.resolve()} | {frame.shape[1]}x{frame.shape[0]} | "
        f"proc_ms={elapsed_ms:.1f} | {summarize_detections(result)}"
    )
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), flush=True)

    if show_preview:
        annotated = vision.draw_detections(frame, result)
        cv2.namedWindow("Street Rush AI — Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Street Rush AI — Image", annotated)
        announce("Pressione qualquer tecla para fechar o preview.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_diag_capture(
    monitor_index: int,
    *,
    captures_dir: Optional[Path] = None,
    args: Optional[argparse.Namespace] = None,
) -> int:
    """Teste controlado: 1 frame REAL do MSS → salva PNG → encerra.

    Não usa imagens sintéticas. Se o MSS falhar neste ambiente, reporta o erro.
    Cada chamada cria uma sessão exclusiva sob ``captures_dir``/``CAPTURES_DIR``.
    """
    if args is None:
        args = build_parser().parse_args(
            ["--diag-capture", "--monitor", str(monitor_index)]
        )
    paths, meta = open_capture_session(args, captures_parent=captures_dir)
    out_dir = paths.frames_dir
    session = RunSession(captures_dir=out_dir)
    startup_delay = float(getattr(args, "startup_delay", STARTUP_DELAY_SEC))

    announce("=== DIAG-CAPTURE: teste controlado com frame REAL do MSS ===")
    announce(f"Sessão: {paths.session_id}")
    announce(f"frames → {paths.frames_dir.resolve()}")
    announce(f"debug  → {paths.debug_dir.resolve()}")
    announce_path_diagnostics(out_dir)
    announce(f"pid={os.getpid()} | python={sys.executable}")
    wait_startup_delay(startup_delay)

    last_resolution: Optional[Tuple[int, int]] = None
    try:
        with ScreenCapture(monitor_index=monitor_index) as capture:
            mon = capture.monitor
            announce(
                f"Monitor selecionado: índice={mon.index} | "
                f"resolução={mon.resolution_label} | origem=({mon.left},{mon.top})"
            )
            announce("Solicitando 1 frame ao MSS…")
            frame = capture.grab()
            session.frames_captured = 1
            session.last_fps = capture.fps
            h, w = frame.shape[:2]
            last_resolution = (w, h)
            announce(
                f"Frame recebido | número=1 | shape={w}x{h} | "
                f"dtype={frame.dtype} | mean_bgr={frame.mean(axis=(0,1)).round(2).tolist()}"
            )
            session.frames_processed = 1
            path = save_frame(frame, 1, out_dir, reason="diag-capture")
            session.images_saved = 1
            session.saved_paths.append(path)
            announce("DIAG-CAPTURE concluído com sucesso (frame real do MSS).")
            print_run_summary(session, reason="diag-capture")
            finalize_metadata(
                meta,
                paths,
                frames_saved=session.images_saved,
                frames_captured=session.frames_captured,
                frames_processed=session.frames_processed,
                resolution=last_resolution,
                fps_avg=session.avg_fps,
            )
            return 0
    except ScreenCaptureError as exc:
        session.capture_errors += 1
        announce(f"ERRO FATAL DE CAPTURA (MSS): {exc}")
        print_run_summary(session, reason=f"diag-capture falhou: {exc}")
        finalize_metadata(
            meta,
            paths,
            frames_saved=session.images_saved,
            frames_captured=session.frames_captured,
            frames_processed=session.frames_processed,
            resolution=last_resolution,
            fps_avg=session.avg_fps,
        )
        return 1
    except OSError as exc:
        session.save_errors += 1
        announce(f"ERRO AO SALVAR frame real: {exc}")
        print_run_summary(session, reason=f"diag-capture gravação falhou: {exc}")
        finalize_metadata(
            meta,
            paths,
            frames_saved=session.images_saved,
            frames_captured=session.frames_captured,
            frames_processed=session.frames_processed,
            resolution=last_resolution,
            fps_avg=session.avg_fps,
        )
        return 1


def run_capture_loop(
    args: argparse.Namespace,
    *,
    captures_dir: Optional[Path] = None,
) -> int:
    """Loop contínuo: captura integral + visão + salvamento em sessão exclusiva."""
    vision = VisionAnalyzer()
    tracker = GameStateTracker()
    paths, meta = open_capture_session(args, captures_parent=captures_dir)
    out_dir = paths.frames_dir

    session = RunSession(captures_dir=out_dir)
    preview = bool(args.preview)
    last_stats_log = 0.0
    last_save = 0.0
    exit_reason = "concluído"
    exit_code = 0
    save_every = int(args.save_every)
    save_interval = float(args.save_interval)
    trace = bool(getattr(args, "trace_frames", True))
    last_resolution: Optional[Tuple[int, int]] = None
    root_sentinel = SAVE_REQUEST_FILE
    session_sentinel = paths.session_dir / ".save_now"

    startup_delay = float(getattr(args, "startup_delay", STARTUP_DELAY_SEC))
    announce("CAPTURANDO MONITOR… (loop contínuo — encerra só com Ctrl+C)")
    announce(f"Sessão exclusiva: {paths.session_id}")
    announce(f"frames → {paths.frames_dir.resolve()}")
    announce(f"metadata → {paths.metadata_path.resolve()}")
    announce(f"debug  → {paths.debug_dir.resolve()}")
    announce_path_diagnostics(out_dir)
    announce(f"pid={os.getpid()} | python={sys.executable}")
    announce(
        f"Argumentos interpretados: save_every={save_every!r} "
        f"(type={type(save_every).__name__}) | save_interval={save_interval!r} | "
        f"monitor={args.monitor} | max_frames={args.max_frames} | "
        f"startup_delay={startup_delay:g}"
    )
    announce(
        f"Sentinela de save: {root_sentinel.resolve()} "
        f"(ou {session_sentinel.resolve()})"
    )
    if save_every > 0:
        announce(
            f"Salvamento automático ATIVO: a cada {save_every} frames processados "
            f"(primeiro PNG esperado no frame {save_every})"
        )
    elif save_interval > 0:
        announce(f"Salvamento automático ATIVO: a cada {save_interval:.1f}s")
    else:
        announce(
            "AVISO CRÍTICO: save_every=0 e save_interval=0 — "
            "o loop vai rodar para sempre SEM gravar PNG. "
            "Use: python -m src.main --save-every 30"
        )
    # Espera ANTES de abrir MSS/grab — nenhum frame processado ainda.
    wait_startup_delay(startup_delay)

    try:
        with ScreenCapture(monitor_index=args.monitor) as capture:
            mon = capture.monitor
            announce(
                f"Monitor selecionado: índice={mon.index} | "
                f"resolução={mon.resolution_label} | origem=({mon.left},{mon.top}) | "
                f"preview={'on' if preview else 'off'}"
            )
            announce(
                "Alt+Tab para o Last War — a captura continua em segundo plano."
            )

            while True:
                try:
                    frame = capture.grab()
                except ScreenCaptureError as exc:
                    session.capture_errors += 1
                    announce(f"ERRO DE CAPTURA: {exc}")
                    time.sleep(0.05)
                    if args.max_frames and session.frames_captured >= args.max_frames:
                        exit_reason = f"limite de frames ({args.max_frames})"
                        break
                    continue

                session.frames_captured += 1
                session.last_fps = capture.fps
                h, w = frame.shape[:2]
                last_resolution = (w, h)
                if trace:
                    announce(
                        f"Frame recebido | número={session.frames_captured} | "
                        f"shape={w}x{h} | mss_fps={capture.fps:.1f}"
                    )

                t0 = time.perf_counter()
                try:
                    result = vision.analyze(frame)
                    state = tracker.update(frame, result, fps=capture.fps)
                except Exception as exc:  # noqa: BLE001 — não engolir sem log
                    announce(f"ERRO NA VISÃO/ESTADO: {exc}")
                    logger.exception("Falha ao processar frame %d", session.frames_captured)
                    continue
                capture.stats.note_process_time(time.perf_counter() - t0)
                session.frames_processed += 1

                frames = session.frames_processed
                now = time.time()
                sentinel = consume_save_request(root_sentinel) or consume_save_request(
                    session_sentinel
                )
                do_save, save_reason = should_save_frame(
                    frames=frames,
                    save_every=save_every,
                    save_interval=save_interval,
                    last_save_ts=last_save,
                    now=now,
                    sentinel=sentinel,
                )

                if trace or do_save:
                    announce(
                        f"Condição de salvamento | frame_processado={frames} | "
                        f"salvar={do_save} | motivo={save_reason}"
                    )

                if do_save:
                    try:
                        path = save_frame(
                            frame, frames, out_dir, reason=save_reason
                        )
                        session.images_saved += 1
                        session.saved_paths.append(path)
                        last_save = now
                    except OSError as exc:
                        session.save_errors += 1
                        announce(f"ERRO AO SALVAR FRAME: {exc}")
                        logger.exception("Falha de gravação no frame %d", frames)

                if args.stats_interval > 0 and (now - last_stats_log) >= args.stats_interval:
                    log_stats(
                        capture,
                        summarize_detections(result),
                        session,
                        dump_state=args.dump_state,
                        state_dict=state.to_dict() if args.dump_state else None,
                    )
                    capture.stats.reset_process_average()
                    last_stats_log = now

                if preview:
                    annotated = vision.draw_detections(frame, result)
                    annotated = capture.draw_overlay(annotated)
                    key = capture.show_preview(annotated)
                    if key == KEY_QUIT:
                        exit_reason = "tecla q no preview"
                        break
                    if key == KEY_SAVE_FRAME:
                        try:
                            path = save_frame(
                                frame, frames, out_dir, reason="tecla s"
                            )
                            session.images_saved += 1
                            session.saved_paths.append(path)
                            last_save = now
                        except OSError as exc:
                            session.save_errors += 1
                            announce(f"ERRO AO SALVAR FRAME: {exc}")

                if args.max_frames and session.frames_processed >= args.max_frames:
                    exit_reason = f"limite de frames ({args.max_frames})"
                    break

    except ScreenCaptureError as exc:
        exit_code = 1
        exit_reason = f"falha ao iniciar/usar MSS: {exc}"
        announce(f"ERRO FATAL DE CAPTURA: {exc}")
    except KeyboardInterrupt:
        exit_reason = "Ctrl+C (KeyboardInterrupt)"
        announce("Interrompido pelo usuário (Ctrl+C).")
    finally:
        if preview:
            ScreenCapture.destroy_windows()
        finalize_metadata(
            meta,
            paths,
            frames_saved=session.images_saved,
            frames_captured=session.frames_captured,
            frames_processed=session.frames_processed,
            resolution=last_resolution,
            fps_avg=session.avg_fps,
            parameters=capture_run_parameters(args),
        )
        announce(f"metadata atualizado: {paths.metadata_path.resolve()}")
        print_run_summary(session, reason=exit_reason)

    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_sessions:
        return print_sessions_list()

    frames_dir, debug_root, _session_paths = resolve_analysis_paths(args)

    if args.render_debug:
        from src.debug_render import run_debug_cli

        hsv_dir = debug_root / "hsv" if _session_paths else debug_root
        return run_debug_cli(captures_dir=frames_dir, debug_dir=hsv_dir)

    if args.render_roi_debug:
        from src.roi_debug import run_roi_debug_cli

        roi_dir = debug_root / "roi" if _session_paths else (debug_root / "roi")
        return run_roi_debug_cli(captures_dir=frames_dir, output_dir=roi_dir)

    if args.render_character_debug:
        from src.character_debug import run_character_debug_cli

        char_dir = debug_root / "character" if _session_paths else (debug_root / "character")
        return run_character_debug_cli(captures_dir=frames_dir, out_dir=char_dir)

    if args.image:
        return analyze_image(Path(args.image), show_preview=bool(args.preview))

    if args.diag_capture:
        return run_diag_capture(args.monitor, args=args)

    return run_capture_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
