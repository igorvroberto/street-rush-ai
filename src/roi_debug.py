"""Geração de debug da ROI da pista (máscara, overlay, detecções).

Preserva frames originais. Não altera HSV.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from src.config import (
    BOX_COLORS,
    CAPTURES_DIR,
    ROI_CORE_NORMALIZED,
    ROI_DEBUG_DIR,
    ROI_REGION_COLORS,
    ROI_SOFT_NORMALIZED,
)
from src.debug_render import list_real_capture_frames, save_debug_image
from src.roi import (
    RoiTaggedDetection,
    count_by_class,
    count_by_region,
    normalized_to_pixels,
    tag_vision_result,
)
from src.vision import VisionAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class FrameRoiReport:
    source: str
    before: dict[str, int]
    after_playable: dict[str, int]  # core + soft
    by_region: dict[str, dict[str, int]]
    total_before: int
    total_playable: int
    total_outside: int
    outputs: dict[str, str]


def render_roi_mask_image(
    height: int,
    width: int,
    core: np.ndarray,
    soft_only: np.ndarray,
) -> np.ndarray:
    """Máscara colorida: núcleo verde, soft âmbar, fora preto."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[soft_only > 0] = (0, 160, 255)  # soft
    img[core > 0] = (0, 220, 0)  # core por cima
    # Legenda
    cv2.putText(img, "core", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
    cv2.putText(img, "soft", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 160, 255), 2)
    cv2.putText(img, "outside", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
    return img


def render_roi_overlay(
    frame: np.ndarray,
    core: np.ndarray,
    soft_only: np.ndarray,
) -> np.ndarray:
    """Contornos do núcleo/soft sobre o frame original."""
    out = frame.copy()
    tint = out.copy()
    tint[soft_only > 0] = (0, 180, 255)
    tint[core > 0] = (0, 220, 0)
    out = cv2.addWeighted(tint, 0.25, out, 0.75, 0)
    core_pts = normalized_to_pixels(ROI_CORE_NORMALIZED, frame.shape[1], frame.shape[0])
    soft_pts = normalized_to_pixels(ROI_SOFT_NORMALIZED, frame.shape[1], frame.shape[0])
    cv2.polylines(out, [soft_pts], True, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.polylines(out, [core_pts], True, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(
        out,
        "ROI core (green) + soft (amber)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def render_roi_detections(
    frame: np.ndarray,
    tagged: Sequence[RoiTaggedDetection],
    before_counts: dict[str, int],
    region_counts: dict[str, dict[str, int]],
) -> np.ndarray:
    """Boxes HSV com cor da região + classe; painel before/after."""
    out = frame.copy()
    for item in tagged:
        det = item.detection
        region_color = ROI_REGION_COLORS.get(item.track_region, (255, 255, 255))
        class_color = BOX_COLORS.get(det.label, (255, 255, 255))
        x, y, bw, bh = det.bbox
        # Borda externa = região; interna = classe
        cv2.rectangle(out, (x, y), (x + bw, y + bh), region_color, 3)
        cv2.rectangle(out, (x + 2, y + 2), (x + bw - 2, y + bh - 2), class_color, 1)
        cx, cy = int(item.center[0]), int(item.center[1])
        cv2.circle(out, (cx, cy), 4, region_color, -1, cv2.LINE_AA)
        label = (
            f"{det.label}/{item.track_region} "
            f"c={item.center_region} "
            f"ic={item.intersection_core:.2f} "
            f"ip={item.intersection_playable:.2f}"
        )
        cv2.putText(
            out,
            label,
            (x, max(14, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            region_color,
            1,
            cv2.LINE_AA,
        )

    playable = {
        k: region_counts["core"][k] + region_counts["soft"][k]
        for k in before_counts
    }
    lines = [
        "Street Rush AI — ROI debug",
        (
            f"BEFORE  char={before_counts['character']} truck={before_counts['truck']} "
            f"barrier={before_counts['barrier']} coin={before_counts['coin']}"
        ),
        (
            f"PLAYABLE char={playable['character']} truck={playable['truck']} "
            f"barrier={playable['barrier']} coin={playable['coin']}"
        ),
        (
            f"core={sum(region_counts['core'].values())} "
            f"soft={sum(region_counts['soft'].values())} "
            f"outside={sum(region_counts['outside'].values())}"
        ),
    ]
    panel_h = 20 + 20 * len(lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (min(720, out.shape[1]), panel_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
    y = 18
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return out


def process_frame_roi(
    frame: np.ndarray,
    source_name: str,
    *,
    analyzer: Optional[VisionAnalyzer] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FrameRoiReport, List[RoiTaggedDetection]]:
    """Analisa um frame e devolve as 3 imagens de debug + relatório."""
    vision = analyzer or VisionAnalyzer()
    h, w = frame.shape[:2]
    result = vision.analyze(frame)
    tagged, core, soft_only, _playable = tag_vision_result(result, w, h)
    before = count_by_class(result.all_objects)
    by_region = count_by_region(tagged)
    playable_counts = {
        k: by_region["core"][k] + by_region["soft"][k] for k in before
    }
    mask_img = render_roi_mask_image(h, w, core, soft_only)
    overlay_img = render_roi_overlay(frame, core, soft_only)
    det_img = render_roi_detections(frame, tagged, before, by_region)
    report = FrameRoiReport(
        source=source_name,
        before=before,
        after_playable=playable_counts,
        by_region=by_region,
        total_before=sum(before.values()),
        total_playable=sum(playable_counts.values()),
        total_outside=sum(by_region["outside"].values()),
        outputs={},
    )
    return mask_img, overlay_img, det_img, report, tagged


def render_roi_debug_for_captures(
    *,
    captures_dir: Path = CAPTURES_DIR,
    output_dir: Path = ROI_DEBUG_DIR,
    sources: Optional[Sequence[Path]] = None,
) -> tuple[List[FrameRoiReport], Path]:
    """Gera debug ROI para todos os frames reais e grava relatório JSON."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = list(sources) if sources is not None else list_real_capture_frames(captures_dir)
    analyzer = VisionAnalyzer()
    reports: List[FrameRoiReport] = []

    for src in inputs:
        frame = cv2.imread(str(src))
        if frame is None:
            logger.error("Falha ao ler %s", src)
            continue
        # Garantir que o original não é sobrescrito: só lemos.
        mask_img, overlay_img, det_img, report, _tagged = process_frame_roi(
            frame, src.name, analyzer=analyzer
        )
        stem = src.stem
        paths = {
            "mask": output_dir / f"{stem}_roi_mask.png",
            "overlay": output_dir / f"{stem}_roi_overlay.png",
            "detections": output_dir / f"{stem}_roi_detections.png",
        }
        save_debug_image(mask_img, paths["mask"])
        save_debug_image(overlay_img, paths["overlay"])
        save_debug_image(det_img, paths["detections"])
        report.outputs = {k: str(v.resolve()) for k, v in paths.items()}
        reports.append(report)
        logger.info(
            "%s before=%s playable=%s outside=%d",
            src.name,
            report.before,
            report.after_playable,
            report.total_outside,
        )

    report_path = output_dir / "roi_report.json"
    payload = [asdict(r) for r in reports]
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return reports, report_path


def run_roi_debug_cli(
    *,
    captures_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> int:
    captures_dir = (captures_dir or CAPTURES_DIR).resolve()
    output_dir = (output_dir or ROI_DEBUG_DIR).resolve()
    print("[street-rush-ai] ROI debug", flush=True)
    print(f"[street-rush-ai] capturas: {captures_dir}", flush=True)
    print(f"[street-rush-ai] saída   : {output_dir}", flush=True)
    frames = list_real_capture_frames(captures_dir)
    print(f"[street-rush-ai] frames reais: {len(frames)}", flush=True)
    reports, report_path = render_roi_debug_for_captures(
        captures_dir=captures_dir, output_dir=output_dir
    )
    print(f"[street-rush-ai] relatório: {report_path}", flush=True)
    for r in reports:
        print(
            f"[street-rush-ai] {r.source} | before={r.total_before} "
            f"playable={r.total_playable} outside={r.total_outside} "
            f"| {r.before} -> {r.after_playable}",
            flush=True,
        )
    return 0 if reports or not frames else 1


__all__ = [
    "FrameRoiReport",
    "process_frame_roi",
    "render_roi_debug_for_captures",
    "run_roi_debug_cli",
]
