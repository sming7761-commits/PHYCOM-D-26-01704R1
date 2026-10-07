"""Protocol-grounded reconstruction of the compact PhyGuard CP-OFDM simulator.

This is a new implementation derived from the submitted manuscript. It is not
claimed to be the unavailable original code. See docs/RECONSTRUCTION_POLICY.md.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.special import j0

KPI_NAMES = [
    "RSRP_dB", "RSSI_dB", "SINR_dB", "EVM", "BER", "PER_proxy",
    "CQI", "MCS", "goodput_norm", "rho_H",
]

# 3GPP TR 38.901 TDL normalized delays and powers. These arrays are isolated
# here so they can be replaced if a later standards-table audit finds a mismatch.
TDL_PROFILES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "TDL-A": (
        np.array([0.0000,0.3819,0.4025,0.5868,0.4610,0.5375,0.6708,0.5750,
                  0.7618,1.5375,1.8978,2.2242,2.1718,2.4942,2.5119,3.0582,
                  4.0810,4.4579,4.5695,4.7966,5.0066,5.3043,9.6586]),
        np.array([-13.4,0.0,-2.2,-4.0,-6.0,-8.2,-9.9,-10.5,-7.5,-15.9,-6.6,
                  -16.7,-12.4,-15.2,-10.8,-11.3,-12.7,-16.2,-18.3,-18.9,
                  -16.6,-19.9,-29.7]),
    ),
    "TDL-B": (
        np.array([0.0000,0.1072,0.2155,0.2095,0.2870,0.2986,0.3752,0.5055,
                  0.3681,0.3697,0.5700,0.5283,1.1021,1.2756,1.5474,1.7842,
                  2.0169,2.8294,3.0219,3.6187,4.1067,4.2790,4.7834]),
        np.array([0.0,-2.2,-4.0,-3.2,-9.8,-1.2,-3.4,-5.2,-7.6,-3.0,-8.9,
                  -9.0,-4.8,-5.7,-7.5,-1.9,-7.6,-12.2,-9.8,-11.4,-14.9,
                  -9.2,-11.3]),
    ),
    "TDL-C": (
        np.array([0.0000,0.2099,0.2219,0.2329,0.2176,0.6366,0.6448,0.6560,
                  0.6584,0.7935,0.8213,0.9336,1.2285,1.3083,2.1704,2.7105,
                  4.2589,4.6003,5.4902,5.6077,6.3065,6.6374,7.0427,8.6523]),
        np.array([-4.4,-1.2,-3.5,-5.2,-2.5,0.0,-2.2,-3.9,-7.4,-7.1,-10.7,
                  -11.1,-5.1,-6.8,-8.7,-13.2,-13.9,-13.9,-15.8,-17.1,-16.0,
                  -15.7,-21.6,-22.8]),
    ),
}

# Reconstructed compact four-level adaptation table. The manuscript did not
# preserve its exact thresholds/rates, so this is explicitly a reconstruction choice.
MCS_TABLE = [
    {"mcs": 0, "threshold_db": -2.0, "mod": "QPSK", "efficiency": 1.0},
    {"mcs": 1, "threshold_db": 4.0,  "mod": "QPSK", "efficiency": 2.0},
    {"mcs": 2, "threshold_db": 10.0, "mod": "16QAM", "efficiency": 4.0},
    {"mcs": 3, "threshold_db": 17.0, "mod": "64QAM", "efficiency": 6.0},
]
MAX_EFFICIENCY = max(row["efficiency"] for row in MCS_TABLE)


@dataclass(frozen=True)
class SequenceSpec:
    regime: str
    label: str
    severity: str
    sample_index: int
    seed: int


def db_to_lin(x_db: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(x_db) / 10.0)


def lin_to_db(x: float | np.ndarray, floor: float = 1e-12) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x), floor))


def qam_constellation(name: str) -> np.ndarray:
    if name == "QPSK":
        pts = np.array([-1-1j, -1+1j, 1-1j, 1+1j], dtype=np.complex128)
    elif name == "16QAM":
        a = np.array([-3, -1, 1, 3])
        pts = np.array([x + 1j*y for x in a for y in a], dtype=np.complex128)
    elif name == "64QAM":
        a = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        pts = np.array([x + 1j*y for x in a for y in a], dtype=np.complex128)
    else:
        raise ValueError(f"Unsupported modulation: {name}")
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def random_qam(name: str, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, int]:
    const = qam_constellation(name)
    idx = rng.integers(0, len(const), size=n, endpoint=False)
    return const[idx], idx, int(np.log2(len(const)))


def hard_demap(symbols: np.ndarray, name: str) -> np.ndarray:
    const = qam_constellation(name)
    d2 = np.abs(symbols[:, None] - const[None, :]) ** 2
    return np.argmin(d2, axis=1)


def symbol_bit_errors(tx_idx: np.ndarray, rx_idx: np.ndarray, bits_per_symbol: int) -> Tuple[int, np.ndarray]:
    xor = np.bitwise_xor(tx_idx.astype(np.uint64), rx_idx.astype(np.uint64))
    per_symbol = np.zeros_like(xor, dtype=np.int16)
    for shift in range(bits_per_symbol):
        per_symbol += ((xor >> shift) & 1).astype(np.int16)
    return int(per_symbol.sum()), per_symbol


def supported_mcs(sinr_db: float) -> int:
    m = 0
    for row in MCS_TABLE:
        if sinr_db >= row["threshold_db"]:
            m = row["mcs"]
    return m


def cqi_from_sinr(sinr_db: float) -> float:
    # Transparent monotonic proxy, not standardized NR CQI.
    return float(np.clip(np.floor((sinr_db + 5.0) / 1.7), 0, 15))


def active_subcarrier_indices(nfft: int = 64, n_active: int = 52) -> np.ndarray:
    half = n_active // 2
    # Centered bins excluding DC, returned in fftshift coordinates.
    return np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])


def make_channel_state(profile: str, rng: np.random.Generator) -> np.ndarray:
    _, p_db = TDL_PROFILES[profile]
    p = db_to_lin(p_db)
    p = p / p.sum()
    return (rng.normal(size=len(p)) + 1j * rng.normal(size=len(p))) * np.sqrt(p / 2.0)


def evolve_channel(state: np.ndarray, profile: str, rho: float, rng: np.random.Generator) -> np.ndarray:
    _, p_db = TDL_PROFILES[profile]
    p = db_to_lin(p_db)
    p = p / p.sum()
    innovation = (rng.normal(size=len(p)) + 1j * rng.normal(size=len(p))) * np.sqrt(p / 2.0)
    return rho * state + np.sqrt(max(0.0, 1.0 - rho**2)) * innovation


def frequency_response(state: np.ndarray, profile: str, rms_delay_ns: float,
                       sc_spacing_hz: float, active_bins: np.ndarray) -> np.ndarray:
    tau_norm, _ = TDL_PROFILES[profile]
    delays_s = tau_norm * rms_delay_ns * 1e-9
    f = active_bins * sc_spacing_hz
    return np.sum(state[:, None] * np.exp(-1j * 2*np.pi * delays_s[:, None] * f[None, :]), axis=0)


def frame_rho(h_now: np.ndarray, h_prev: np.ndarray) -> float:
    denom = np.linalg.norm(h_now) * np.linalg.norm(h_prev)
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.abs(np.vdot(h_prev, h_now)) / denom, 0.0, 1.0))


def simulate_sequence(spec: SequenceSpec, protocol: dict) -> Tuple[np.ndarray, dict]:
    rng = np.random.default_rng(spec.seed)
    w = protocol["waveform"]
    regime_cfg = protocol["regimes"][spec.regime]
    interventions = protocol["interventions"]

    T = int(w["sequence_frames"])
    n_sym = int(w["ofdm_symbols_per_frame"])
    n_active = int(w["active_subcarriers"])
    active_bins = active_subcarrier_indices(int(w["fft_size"]), n_active)

    event_start = int(rng.integers(w["event_start_range_inclusive"][0], w["event_start_range_inclusive"][1] + 1))
    event_duration = int(rng.integers(w["event_duration_range_inclusive"][0], w["event_duration_range_inclusive"][1] + 1))
    event_end = min(T, event_start + event_duration)

    snr_db = float(rng.uniform(*regime_cfg["snr_db_range"]))
    base_fd = float(rng.uniform(*regime_cfg["doppler_hz_range"]))
    noise_power_base = float(1.0 / db_to_lin(snr_db))
    profile = regime_cfg["tdl"]
    rms_delay_ns = float(regime_cfg["rms_delay_ns"])

    h_state = make_channel_state(profile, rng)
    g_state = make_channel_state(profile, rng)
    h_prev = frequency_response(h_state, profile, rms_delay_ns, w["subcarrier_spacing_hz"], active_bins)
    # Initial estimate is close to the first channel, with small estimation noise.
    est_noise_scale = 0.004 + 0.012 / np.sqrt(max(db_to_lin(snr_db), 1e-6))
    h_est = h_prev + est_noise_scale * (rng.normal(size=n_active) + 1j*rng.normal(size=n_active)) / np.sqrt(2)
    filtered_sinr_db = snr_db
    mcs_current = supported_mcs(filtered_sinr_db)
    mcs_before_event = mcs_current

    out = np.zeros((T, len(KPI_NAMES)), dtype=np.float64)

    for t in range(T):
        active_event = event_start <= t < event_end and spec.label not in {"normal"}
        fd = base_fd
        desired_gain = 1.0
        noise_power = noise_power_base
        js_db = None

        if active_event and spec.label == "interference":
            js_db = float(interventions["interference_js_db"][spec.severity])
        elif active_event and spec.label == "blockage":
            att_db = float(interventions["blockage_attenuation_db"][spec.severity])
            desired_gain = float(np.sqrt(db_to_lin(-att_db)))
        elif active_event and spec.label == "mobility":
            fd = float(interventions["mobility_doppler_hz"][spec.severity])
        elif active_event and spec.label == "adaptation_mismatch":
            loss_db = float(interventions["adaptation_quality_loss_db"][spec.severity])
            # Reconstructed as additional impairment/noise, preserving desired power.
            noise_power = noise_power_base * float(db_to_lin(loss_db))

        rho = float(j0(2*np.pi * fd * float(w["frame_duration_s"])))
        rho = float(np.clip(rho, -0.999999, 0.999999))
        h_state = evolve_channel(h_state, profile, rho, rng)
        # Interferer remains independently time-varying at the regime Doppler.
        rho_g = float(np.clip(j0(2*np.pi * base_fd * float(w["frame_duration_s"])), -0.999999, 0.999999))
        g_state = evolve_channel(g_state, profile, rho_g, rng)
        h_true = desired_gain * frequency_response(h_state, profile, rms_delay_ns, w["subcarrier_spacing_hz"], active_bins)
        g_true = frequency_response(g_state, profile, rms_delay_ns, w["subcarrier_spacing_hz"], active_bins)

        # One-frame-delayed, low-pass channel estimate. The manuscript did not retain
        # the exact coefficient, so beta=0.70 is a locked reconstruction choice.
        delayed_obs = h_prev + est_noise_scale * (rng.normal(size=n_active) + 1j*rng.normal(size=n_active)) / np.sqrt(2)
        beta = 0.18
        h_est = beta * h_est + (1.0 - beta) * delayed_obs

        # Link adaptation uses a filtered quality estimate. During mismatch, the prior
        # MCS is held for the specified number of event frames.
        desired_power_est = float(np.mean(np.abs(h_true) ** 2))
        true_sinr_db = float(lin_to_db(desired_power_est / noise_power))
        if js_db is not None:
            int_power_target = desired_power_est * float(db_to_lin(js_db))
            true_sinr_db = float(lin_to_db(desired_power_est / (noise_power + int_power_target)))
        filtered_sinr_db = 0.78 * filtered_sinr_db + 0.22 * true_sinr_db
        support = supported_mcs(filtered_sinr_db)
        if t == event_start:
            mcs_before_event = mcs_current
        if active_event and spec.label == "adaptation_mismatch":
            hold = int(interventions["adaptation_mcs_hold_frames"][spec.severity])
            if t - event_start < hold:
                mcs_current = mcs_before_event
            else:
                mcs_current = support
        else:
            mcs_current = support
        mod_name = MCS_TABLE[mcs_current]["mod"]
        efficiency = MCS_TABLE[mcs_current]["efficiency"]

        n_symbols_total = n_sym * n_active
        tx, tx_idx, bps = random_qam(mod_name, n_symbols_total, rng)
        tx = tx.reshape(n_sym, n_active)
        tx_idx = tx_idx.reshape(n_sym, n_active)

        desired = h_true[None, :] * tx
        interference = np.zeros_like(desired)
        int_power = 0.0
        if js_db is not None:
            int_mod = str(rng.choice(w["interferer_modulations"]))
            itx, _, _ = random_qam(int_mod, n_symbols_total, rng)
            itx = itx.reshape(n_sym, n_active)
            raw_i = g_true[None, :] * itx
            raw_power = float(np.mean(np.abs(raw_i) ** 2))
            target = desired_power_est * float(db_to_lin(js_db))
            scale = np.sqrt(target / max(raw_power, 1e-12))
            interference = raw_i * scale
            int_power = float(np.mean(np.abs(interference) ** 2))

        noise = np.sqrt(noise_power/2.0) * (rng.normal(size=desired.shape) + 1j*rng.normal(size=desired.shape))
        y = desired + interference + noise
        x_eq = y / np.where(np.abs(h_est[None, :]) < 1e-8, 1e-8 + 0j, h_est[None, :])

        rx_idx = hard_demap(x_eq.reshape(-1), mod_name).reshape(n_sym, n_active)
        bit_errors, per_symbol_errors = symbol_bit_errors(tx_idx.reshape(-1), rx_idx.reshape(-1), bps)
        total_bits = n_symbols_total * bps
        ber = bit_errors / max(total_bits, 1)

        # Exact binary-label comparison for the reconstructed uncoded packet proxy.
        tx_flat = tx_idx.reshape(-1).astype(np.uint64)
        rx_flat = rx_idx.reshape(-1).astype(np.uint64)
        shifts = np.arange(bps - 1, -1, -1, dtype=np.uint64)
        tx_bits = ((tx_flat[:, None] >> shifts[None, :]) & 1).reshape(-1)
        rx_bits = ((rx_flat[:, None] >> shifts[None, :]) & 1).reshape(-1)
        bit_err_stream = tx_bits != rx_bits
        packet_bits = int(w["packet_surrogate_bits"])
        n_packets = max(1, len(bit_err_stream) // packet_bits)
        trimmed = bit_err_stream[:n_packets * packet_bits].reshape(n_packets, packet_bits)
        per_proxy = float(np.mean(np.any(trimmed, axis=1)))

        evm = float(np.sqrt(np.mean(np.abs(x_eq - tx) ** 2) / max(np.mean(np.abs(tx) ** 2), 1e-12)))
        desired_power = float(np.mean(np.abs(desired) ** 2))
        rssi_power = float(np.mean(np.abs(y) ** 2))
        sinr_lin = desired_power / max(noise_power + int_power, 1e-12)
        sinr_db_meas = float(lin_to_db(sinr_lin))
        cqi = cqi_from_sinr(sinr_db_meas)
        goodput = (efficiency / MAX_EFFICIENCY) * (1.0 - per_proxy)
        if active_event and spec.label == "nonphysical_goodput":
            goodput = min(goodput, float(interventions["nonphysical_goodput_cap"][spec.severity]))
        rho_h = frame_rho(h_true, h_prev)

        out[t] = [
            float(lin_to_db(desired_power)), float(lin_to_db(rssi_power)), sinr_db_meas,
            evm, ber, per_proxy, cqi, float(mcs_current), float(np.clip(goodput, 0, 1)), rho_h,
        ]
        h_prev = h_true.copy()

    metadata = {
        "regime": spec.regime,
        "label": spec.label,
        "severity": spec.severity,
        "sample_index": spec.sample_index,
        "seed": spec.seed,
        "event_start": event_start,
        "event_end": event_end,
        "event_duration": event_end - event_start,
        "baseline_snr_db": snr_db,
        "baseline_doppler_hz": base_fd,
        "tdl": profile,
        "rms_delay_ns": rms_delay_ns,
    }
    return out, metadata


def build_specs(protocol: dict, mode: str, samples_per_cell: int) -> Iterable[SequenceSpec]:
    root = int(protocol["global_seed"])
    labels = protocol["labels"]
    regimes = list(protocol["regimes"].keys())
    severities = protocol["severities"]
    idx = 0
    if mode == "smoke":
        # Severe events maximize directional visibility in the first audit.
        for regime in regimes:
            for label in labels:
                sev = "severe"
                for r in range(samples_per_cell):
                    seed = int(np.random.SeedSequence([root, idx, r]).generate_state(1)[0])
                    yield SequenceSpec(regime, label, sev, r, seed)
                idx += 1
    elif mode == "full":
        for regime in regimes:
            for label in labels:
                for severity in severities:
                    for r in range(samples_per_cell):
                        seed = int(np.random.SeedSequence([root, idx, r]).generate_state(1)[0])
                        yield SequenceSpec(regime, label, severity, r, seed)
                    idx += 1
    else:
        raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--samples-per-cell", type=int, default=3,
                    help="Smoke: per regime-label. Full: per regime-label-severity (15 gives 1080).")
    args = ap.parse_args()

    protocol = json.loads(args.protocol.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sequences = []
    rows = []
    for global_idx, spec in enumerate(build_specs(protocol, args.mode, args.samples_per_cell)):
        x, meta = simulate_sequence(spec, protocol)
        meta["global_index"] = global_idx
        sequences.append(x.astype(np.float32))
        rows.append(meta)

    arr = np.stack(sequences)
    np.savez_compressed(args.out_dir / "telemetry.npz", X=arr, kpi_names=np.array(KPI_NAMES))
    pd.DataFrame(rows).to_csv(args.out_dir / "metadata.csv", index=False)
    summary = {
        "mode": args.mode,
        "n_sequences": int(arr.shape[0]),
        "shape": list(arr.shape),
        "finite": bool(np.isfinite(arr).all()),
        "kpi_names": KPI_NAMES,
    }
    (args.out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
