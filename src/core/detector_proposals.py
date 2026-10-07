"""Generate transparent 24-frame interval proposals from three unsupervised detectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from features import build_one

DETECTORS = ["robust_energy", "pca_reconstruction", "isolation_forest"]


def robust_standardize(x: np.ndarray, prefix: int = 24) -> np.ndarray:
    ref = x[:prefix]
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0)
    scale = 1.4826 * mad + 1e-6
    return np.clip((x - med) / scale, -10.0, 10.0)


def sliding_window_start(score: np.ndarray, window: int = 24, earliest: int = 24) -> int:
    latest = len(score) - window
    starts = np.arange(earliest, latest + 1)
    means = np.array([score[s:s+window].mean() for s in starts])
    return int(starts[int(np.argmax(means))])


def proposal_for_sequence(x: np.ndarray, detector: str, seed: int, window: int = 24, global_iforest=None) -> tuple[int, int, np.ndarray]:
    z = robust_standardize(x, prefix=24)
    if detector == "robust_energy":
        score = np.sqrt(np.mean(z * z, axis=1))
    elif detector == "pca_reconstruction":
        n_components = min(5, z.shape[1], 23)
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(z[:24])
        reconstructed = pca.inverse_transform(pca.transform(z))
        score = np.mean((z - reconstructed) ** 2, axis=1)
    elif detector == "isolation_forest":
        if global_iforest is None:
            raise ValueError("global_iforest is required for isolation_forest")
        score = -global_iforest.score_samples(z)
    else:
        raise ValueError(detector)
    start = sliding_window_start(score, window=window, earliest=24)
    return start, start + window, score


def interval_iou(a: int, b: int, c: int, d: int) -> float:
    inter = max(0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    return float(inter / union) if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--detectors", nargs="+", choices=DETECTORS, default=DETECTORS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_dir / "telemetry.npz")
    X = data["X"]
    metadata = pd.read_csv(args.data_dir / "metadata.csv")

    summary = {}
    global_iforest = None
    if "isolation_forest" in args.detectors:
        prefix_frames = np.concatenate([robust_standardize(x, prefix=24)[:24] for x in X], axis=0)
        global_iforest = IsolationForest(
            n_estimators=200,
            max_samples=min(4096, len(prefix_frames)),
            contamination="auto",
            random_state=20260705,
            n_jobs=-1,
        )
        global_iforest.fit(prefix_frames)
    for detector in args.detectors:
        rows, raw_features, physical_features = [], [], []
        score_dir = args.out_dir / detector / "scores"
        score_dir.mkdir(parents=True, exist_ok=True)
        for i, row in metadata.iterrows():
            start, end, score = proposal_for_sequence(X[i], detector, int(row.seed) ^ 0x5A17, args.window, global_iforest=global_iforest)
            raw, physical = build_one(X[i], start, end)
            raw_features.append(raw)
            physical_features.append(physical)
            iou = interval_iou(start, end, int(row.event_start), int(row.event_end))
            rows.append({
                "global_index": int(row.global_index),
                "detector": detector,
                "proposal_start": start,
                "proposal_end": end,
                "event_start": int(row.event_start),
                "event_end": int(row.event_end),
                "iou": iou,
                "label": row.label,
                "severity": row.severity,
                "regime": row.regime,
                "max_frame_score": float(np.max(score)),
                "proposal_mean_score": float(np.mean(score[start:end])),
            })
        detector_dir = args.out_dir / detector
        pd.DataFrame(rows).to_csv(detector_dir / "proposals.csv", index=False)
        raw_arr = np.asarray(raw_features, dtype=np.float32)
        physical_arr = np.asarray(physical_features, dtype=np.float32)
        np.savez_compressed(detector_dir / "features.npz", raw=raw_arr, physical=physical_arr, combined=np.concatenate([raw_arr, physical_arr], axis=1))
        physical_rows = pd.DataFrame(rows)[pd.DataFrame(rows)["label"].isin(["interference", "blockage", "mobility", "adaptation_mismatch"])]
        summary[detector] = {
            "mean_iou_physical": float(physical_rows["iou"].mean()),
            "event_recall_iou_ge_0_3": float((physical_rows["iou"] >= 0.3).mean()),
            "n": int(len(rows)),
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
