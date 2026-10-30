"""Debug e validação do detector de personagem (REAL vs DETECTADO vs HSV legado).

Não altera ROI, HSV legado nem truck/barrier/coin.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from src.character import (
    CharacterDetection,
    CharacterDetector,
    center_distance_norm,
    detection_hits_ground_truth,
    make_character_detector,
)
from src.config import (
    CAPTURES_DIR,
    CHARACTER_DEBUG_DIR,
    CHARACTER_HIT_MAX_DISTANCE,
    CHARACTER_VARIANT_BASELINE,
    CHARACTER_VARIANT_DEFAULT,
    CHARACTER_VARIANT_EXPERIMENTAL,
    ROI_CORE_NORMALIZED,
    ROI_SOFT_NORMALIZED,
)
from src.debug_render import list_real_capture_frames, save_debug_image
from src.roi import build_roi_masks, normalized_to_pixels, point_region
from src.vision import DetectedObject, VisionAnalyzer

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = CHARACTER_DEBUG_DIR / "ground_truth.json"

# BGR
COLOR_REAL = (255, 255, 0)  # ciano-amarelo claro → usar ciano puro
COLOR_REAL_CYAN = (255, 220, 0)
COLOR_DET = (0, 255, 0)  # verde
COLOR_HSV_LEGACY = (255, 0, 255)  # magenta
COLOR_CORE = (0, 220, 0)
COLOR_SOFT = (0, 200, 255)


@dataclass
class FrameCharacterReport:
    source: str
    hit: bool
    gt_center_norm: Optional[Tuple[float, float]]
    gt_expected_region: Optional[str]
    detected: Optional[dict[str, Any]]
    hsv_legacy: Optional[dict[str, Any]]
    distance_to_gt: Optional[float]
    region_ok: Optional[bool]
    false_positive: bool
    false_negative: bool
    output_png: str
    notes: str = ""


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth ausente: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _obj_to_dict(obj: Optional[DetectedObject]) -> Optional[dict[str, Any]]:
    if obj is None:
        return None
    return {
        "bbox": list(obj.bbox),
        "center": [obj.center[0], obj.center[1]],
        "confidence": obj.confidence,
        "label": obj.label,
    }


def _det_to_dict(det: Optional[CharacterDetection], width: int, height: int) -> Optional[dict[str, Any]]:
    if det is None:
        return None
    return {
        "bbox": list(det.bbox),
        "center": [det.center[0], det.center[1]],
        "center_norm": [det.center_norm[0], det.center_norm[1]],
        "confidence": det.confidence,
        "region": det.region,
        "source_cue": det.source_cue,
        "score": det.score,
        "color_cues": det.color_cues,
        "texture_v_std": det.texture_v_std,
        "variant": det.variant,
    }


def render_character_compare(
    frame: np.ndarray,
    *,
    gt_center_norm: Optional[Tuple[float, float]],
    detection: Optional[CharacterDetection],
    hsv_legacy: Optional[DetectedObject],
    title: str = "",
) -> np.ndarray:
    """Overlay: ROI + REAL (ciano) + DETECTADO (verde) + HSV legado (magenta)."""
    out = frame.copy()
    h, w = out.shape[:2]
    core, soft_only, _ = build_roi_masks(w, h)
    tint = out.copy()
    tint[soft_only > 0] = COLOR_SOFT
    tint[core > 0] = COLOR_CORE
    out = cv2.addWeighted(tint, 0.18, out, 0.82, 0)
    soft_pts = normalized_to_pixels(ROI_SOFT_NORMALIZED, w, h)
    core_pts = normalized_to_pixels(ROI_CORE_NORMALIZED, w, h)
    cv2.polylines(out, [soft_pts], True, COLOR_SOFT, 2, cv2.LINE_AA)
    cv2.polylines(out, [core_pts], True, COLOR_CORE, 2, cv2.LINE_AA)

    cv2.putText(
        out,
        "Char: cyan=REAL | green=DETECTED | magenta=HSV legacy",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    if title:
        cv2.putText(
            out,
            title,
            (16, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    if hsv_legacy is not None:
        x, y, bw, bh = hsv_legacy.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), COLOR_HSV_LEGACY, 2)
        cv2.putText(
            out,
            f"HSV legacy {hsv_legacy.confidence:.2f}",
            (x, max(16, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_HSV_LEGACY,
            1,
            cv2.LINE_AA,
        )

    if detection is not None:
        x, y, bw, bh = detection.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), COLOR_DET, 2)
        cx, cy = int(detection.center[0]), int(detection.center[1])
        cv2.drawMarker(out, (cx, cy), COLOR_DET, cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.putText(
            out,
            f"DET {detection.region} conf={detection.confidence:.2f} ({detection.source_cue})",
            (x, max(16, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_DET,
            1,
            cv2.LINE_AA,
        )

    if gt_center_norm is not None:
        gx = int(round(gt_center_norm[0] * (w - 1)))
        gy = int(round(gt_center_norm[1] * (h - 1)))
        cv2.circle(out, (gx, gy), 28, COLOR_REAL_CYAN, 2, cv2.LINE_AA)
        cv2.drawMarker(out, (gx, gy), COLOR_REAL_CYAN, cv2.MARKER_CROSS, 28, 2)
        core_m, soft_m, _ = build_roi_masks(w, h)
        reg = point_region(float(gx), float(gy), core_m, soft_m)
        cv2.putText(
            out,
            f"REAL -> {reg}",
            (gx + 18, gy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_REAL_CYAN,
            2,
            cv2.LINE_AA,
        )

    return out


def evaluate_frame(
    frame: np.ndarray,
    *,
    source_name: str,
    gt_center_norm: Optional[Tuple[float, float]],
    gt_expected_region: Optional[str],
    detector: CharacterDetector,
    vision: VisionAnalyzer,
    notes: str = "",
    use_temporal: bool = False,
) -> tuple[FrameCharacterReport, np.ndarray]:
    """Roda detector + HSV legado e monta relatório/imagem de um frame."""
    h, w = frame.shape[:2]
    detection = detector.detect(frame, use_temporal=use_temporal)
    hsv_result = vision.analyze(frame)
    hsv_legacy = hsv_result.character

    hit = False
    dist: Optional[float] = None
    region_ok: Optional[bool] = None
    false_positive = False
    false_negative = False

    if gt_center_norm is not None:
        hit = detection_hits_ground_truth(
            detection,
            (float(gt_center_norm[0]), float(gt_center_norm[1])),
            max_distance=CHARACTER_HIT_MAX_DISTANCE,
        )
        if detection is not None:
            dist = center_distance_norm(detection.center_norm, (float(gt_center_norm[0]), float(gt_center_norm[1])))
            if gt_expected_region is not None:
                region_ok = detection.region == gt_expected_region
            false_positive = not hit
            false_negative = False
        else:
            false_negative = True
            false_positive = False
    elif detection is not None:
        # Sem GT: não conta como acerto por “região central”.
        false_positive = False

    title = f"{source_name} | hit={hit}"
    if dist is not None:
        title += f" d={dist:.3f}"
    canvas = render_character_compare(
        frame,
        gt_center_norm=gt_center_norm,
        detection=detection,
        hsv_legacy=hsv_legacy,
        title=title,
    )
    report = FrameCharacterReport(
        source=source_name,
        hit=hit,
        gt_center_norm=tuple(gt_center_norm) if gt_center_norm else None,
        gt_expected_region=gt_expected_region,
        detected=_det_to_dict(detection, w, h),
        hsv_legacy=_obj_to_dict(hsv_legacy),
        distance_to_gt=dist,
        region_ok=region_ok,
        false_positive=false_positive,
        false_negative=false_negative,
        output_png="",
        notes=notes,
    )
    return report, canvas


def run_character_debug(
    *,
    captures_dir: Path = CAPTURES_DIR,
    out_dir: Path = CHARACTER_DEBUG_DIR,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    use_temporal: bool = False,
    variant: str = CHARACTER_VARIANT_DEFAULT,
) -> dict[str, Any]:
    """Processa os frames reais, grava PNGs de comparação e relatório JSON."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_frames: dict[str, Any] = {}
    if ground_truth_path is not None and Path(ground_truth_path).is_file():
        gt_doc = load_ground_truth(Path(ground_truth_path))
        gt_frames = gt_doc.get("frames", {})
    else:
        logger.info(
            "Ground truth ausente em %s — gera debug sem métricas de acerto "
            "(não usar detecção como GT automático).",
            ground_truth_path,
        )

    frames = list_real_capture_frames(captures_dir)
    detector = make_character_detector(variant)
    vision = VisionAnalyzer()
    reports: List[FrameCharacterReport] = []

    for path in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Falha ao ler %s", path)
            continue
        meta = gt_frames.get(path.name, {})
        gt_center = meta.get("center_norm")
        gt_tuple = (float(gt_center[0]), float(gt_center[1])) if gt_center else None
        expected_region = meta.get("expected_region")
        notes = str(meta.get("notes", ""))
        if not use_temporal:
            detector.reset_temporal()
        report, canvas = evaluate_frame(
            frame,
            source_name=path.name,
            gt_center_norm=gt_tuple,
            gt_expected_region=expected_region,
            detector=detector,
            vision=vision,
            notes=notes,
            use_temporal=use_temporal,
        )
        out_name = f"{path.stem}_character_detect.png"
        out_path = save_debug_image(canvas, out_dir / out_name)
        report.output_png = str(out_path)
        reports.append(report)

    hits = sum(1 for r in reports if r.hit)
    fps = sum(1 for r in reports if r.false_positive)
    fns = sum(1 for r in reports if r.false_negative)
    with_gt = sum(1 for r in reports if r.gt_center_norm is not None)
    summary = {
        "frames_total": len(reports),
        "frames_with_gt": with_gt,
        "hits": hits,
        "misses": with_gt - hits,
        "false_positives": fps,
        "false_negatives": fns,
        "hit_rate": (hits / with_gt) if with_gt else None,
        "hit_max_distance": CHARACTER_HIT_MAX_DISTANCE,
        "use_temporal": use_temporal,
        "variant": detector.variant,
        "note": (
            "Acerto = distância do centro DETECTADO ao REAL (corpo) ≤ limiar. "
            "Não conta sucesso só por região central. HSV legado só para comparação. "
            "baseline = scoring atual; experimental = anti-painel + multi-cor."
        ),
        "frames": [asdict(r) for r in reports],
    }
    report_path = out_dir / "character_detection_report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary


def run_character_variant_compare(
    *,
    captures_dir: Path = CAPTURES_DIR,
    out_dir: Path = CHARACTER_DEBUG_DIR,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    use_temporal: bool = False,
) -> dict[str, Any]:
    """Roda baseline e experimental lado a lado para comparação offline."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_character_debug(
        captures_dir=captures_dir,
        out_dir=out_dir / "baseline",
        ground_truth_path=ground_truth_path,
        use_temporal=use_temporal,
        variant=CHARACTER_VARIANT_BASELINE,
    )
    experimental = run_character_debug(
        captures_dir=captures_dir,
        out_dir=out_dir / "experimental",
        ground_truth_path=ground_truth_path,
        use_temporal=use_temporal,
        variant=CHARACTER_VARIANT_EXPERIMENTAL,
    )

    base_by_src = {fr["source"]: fr for fr in baseline["frames"]}
    exp_by_src = {fr["source"]: fr for fr in experimental["frames"]}
    sources = sorted(set(base_by_src) | set(exp_by_src))
    per_frame: List[dict[str, Any]] = []
    agree_both_hit = 0
    disagree = 0
    only_baseline = 0
    only_experimental = 0
    for src in sources:
        b = base_by_src.get(src)
        e = exp_by_src.get(src)
        b_det = (b or {}).get("detected")
        e_det = (e or {}).get("detected")
        b_found = b_det is not None
        e_found = e_det is not None
        same_box = False
        if b_found and e_found:
            same_box = list(b_det["bbox"]) == list(e_det["bbox"])
            if (b or {}).get("hit") and (e or {}).get("hit"):
                agree_both_hit += 1
            if bool((b or {}).get("hit")) != bool((e or {}).get("hit")) or not same_box:
                disagree += 1
        elif b_found and not e_found:
            only_baseline += 1
            disagree += 1
        elif e_found and not b_found:
            only_experimental += 1
            disagree += 1
        per_frame.append(
            {
                "source": src,
                "baseline": {
                    "found": b_found,
                    "hit": (b or {}).get("hit"),
                    "detected": b_det,
                },
                "experimental": {
                    "found": e_found,
                    "hit": (e or {}).get("hit"),
                    "detected": e_det,
                },
                "same_bbox": same_box,
            }
        )

    compare = {
        "mode": "compare",
        "frames_total": len(sources),
        "baseline": {
            k: v
            for k, v in baseline.items()
            if k not in {"frames", "report_path"}
        },
        "experimental": {
            k: v
            for k, v in experimental.items()
            if k not in {"frames", "report_path"}
        },
        "agreement": {
            "both_hit_when_gt": agree_both_hit,
            "disagree_or_different_box": disagree,
            "only_baseline_detection": only_baseline,
            "only_experimental_detection": only_experimental,
        },
        "baseline_report_path": baseline.get("report_path"),
        "experimental_report_path": experimental.get("report_path"),
        "frames": per_frame,
        "note": (
            "Compare offline: baseline (atual) vs experimental "
            "(penaliza painéis homogêneos + exige combinação de cores/textura)."
        ),
    }
    report_path = out_dir / "character_variant_compare.json"
    report_path.write_text(json.dumps(compare, indent=2, ensure_ascii=False), encoding="utf-8")
    compare["report_path"] = str(report_path)
    return compare


def _print_debug_summary(summary: dict[str, Any], out_resolved: Path) -> int:
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2, ensure_ascii=False))
    print("--- por frame ---")
    for fr in summary["frames"]:
        det = fr.get("detected")
        det_s = (
            f"reg={det['region']} conf={det['confidence']:.2f} cue={det['source_cue']} "
            f"cues={det.get('color_cues', '?')} v_std={det.get('texture_v_std', 0):.1f} "
            f"c=({det['center_norm'][0]:.3f},{det['center_norm'][1]:.3f})"
            if det
            else "NONE"
        )
        print(
            f"{fr['source']}: hit={fr['hit']} d={fr['distance_to_gt']} "
            f"FP={fr['false_positive']} FN={fr['false_negative']} | {det_s}"
        )
    print(f"Relatório: {summary.get('report_path')}")
    print(f"Imagens: {out_resolved}")
    print(f"Variante: {summary.get('variant', CHARACTER_VARIANT_DEFAULT)}")
    hits = summary["hits"]
    total = summary["frames_with_gt"]
    print(f"Resultado: {hits}/{total} localizações corretas")
    if total == 0:
        return 0 if summary["frames_total"] >= 0 else 1
    return 0 if hits == total else 1


def run_character_debug_cli(
    *,
    captures_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    ground_truth_path: Optional[Path] = None,
    variant: str = CHARACTER_VARIANT_DEFAULT,
) -> int:
    captures = captures_dir or CAPTURES_DIR
    out_resolved = Path(out_dir or CHARACTER_DEBUG_DIR).resolve()
    gt_path = ground_truth_path or GROUND_TRUTH_PATH
    mode = str(variant or CHARACTER_VARIANT_DEFAULT).strip().lower()

    if mode == "compare":
        compare = run_character_variant_compare(
            captures_dir=captures,
            out_dir=out_resolved,
            ground_truth_path=gt_path,
        )
        print(
            json.dumps(
                {k: v for k, v in compare.items() if k != "frames"},
                indent=2,
                ensure_ascii=False,
            )
        )
        print("--- divergências (amostra) ---")
        shown = 0
        for fr in compare["frames"]:
            if fr["same_bbox"] and fr["baseline"]["found"] == fr["experimental"]["found"]:
                if fr["baseline"].get("hit") == fr["experimental"].get("hit"):
                    continue
            b = fr["baseline"]["detected"]
            e = fr["experimental"]["detected"]
            b_s = "NONE" if not b else f"{b['source_cue']}@{b['bbox']}"
            e_s = (
                "NONE"
                if not e
                else f"{e['source_cue']} cues={e.get('color_cues')}@{e['bbox']}"
            )
            print(f"{fr['source']}: baseline={b_s} | experimental={e_s}")
            shown += 1
            if shown >= 25:
                break
        print(f"Compare report: {compare.get('report_path')}")
        print(f"baseline imgs: {out_resolved / 'baseline'}")
        print(f"experimental imgs: {out_resolved / 'experimental'}")
        return 0

    summary = run_character_debug(
        captures_dir=captures,
        out_dir=out_resolved,
        ground_truth_path=gt_path,
        variant=mode,
    )
    return _print_debug_summary(summary, out_resolved)


__all__ = [
    "FrameCharacterReport",
    "evaluate_frame",
    "load_ground_truth",
    "render_character_compare",
    "run_character_debug",
    "run_character_debug_cli",
    "run_character_variant_compare",
]

