import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import sionna
import sionna.phy

from sionna.phy.channel import (
    ApplyOFDMChannel,
    GenerateOFDMChannel,
)
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import BinarySource, Demapper, Mapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    LSChannelEstimator,
    LMMSEEqualizer,
    RemoveNulledSubcarriers,
    ResourceGrid,
    ResourceGridMapper,
)
from sionna.phy.utils import ebnodb2no, expand_to_rank


ROOT = Path("/root/phyguard_revision")
CONTRACT_PATH = ROOT / "contracts/kpi_contract_v1.json"
PLAN_PATH = Path(os.environ.get("PHYGUARD_PLAN_PATH", str(ROOT / "configs/smoke24_plan_v1.json")))
OUTPUT_DIR = Path(os.environ["PHYGUARD_OUTPUT_DIR"])
RESULT_DIR = Path(os.environ["PHYGUARD_RESULT_DIR"])
LOGICAL_GENERATOR_VERSION = "sionna_simo_interference_v1.0.0"

DEVICE = "cuda:0"
PRECISION = "single"

CSI_AGE_FRAMES = 2

NUM_FRAMES = 80
OFDM_SYMBOLS_PER_FRAME = 14

NUM_TX = 1
NUM_STREAMS_PER_TX = 1
NUM_RX = 1
NUM_RX_ANT = 2

FFT_SIZE = 64
SUBCARRIER_SPACING = 30e3
CYCLIC_PREFIX_LENGTH = 16
NUM_GUARD_CARRIERS = [5, 6]
DC_NULL = True
PILOT_SYMBOLS = [2, 11]

BITS_PER_SYMBOL = 2
CODERATE = 0.5
DECODER_ITERATIONS = 15

CARRIER_FREQUENCY_HZ = 3.5e9
DELAY_SPREAD_S = 300e-9

# Normal link: moderate pedestrian mobility.
UT_SPEED_MPS = 3.0

EPS = 1e-12

# The normal operating point is calibrated before generating anomalies.
# Environment variables permit a transparent, reproducible sweep without
# modifying the source code for every candidate point.
NORMAL_EBNO_BASE_DB = float(
    os.environ.get("NORMAL_EBNO_BASE_DB", "12.0")
)
NORMAL_EBNO_SWING_DB = float(
    os.environ.get("NORMAL_EBNO_SWING_DB", "0.5")
)

# Locked mild co-channel interference intervention.
# Target SIR is defined relative to the desired received power.
INTERFERENCE_SIR_DB = float(
    os.environ.get(
        "PHYGUARD_INTERFERENCE_SIR_DB",
        "6.0",
    )
)
INTERFERENCE_RAMP_FRAMES = 4
INTERFERENCE_SEED_OFFSET = 1000003

KPI_NAMES = [
    "rsrp_db",
    "rssi_db",
    "sinr_db",
    "evm_rms",
    "ber_pre_ldpc",
    "bler_post_ldpc",
    "cqi_index",
    "mcs_index",
    "goodput_ratio",
    "channel_correlation",
]


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def select_sample(plan: dict, sample_id: str) -> dict:
    matches = [
        item
        for item in plan["samples"]
        if item["sample_id"] == sample_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {sample_id}, found {len(matches)}"
        )

    return matches[0]


def db10(value: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(torch.clamp(value, min=EPS))


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    sionna.phy.config.device = DEVICE
    sionna.phy.config.precision = PRECISION
    sionna.phy.config.seed = seed


def build_slot_resource_grid() -> ResourceGrid:
    return ResourceGrid(
        num_ofdm_symbols=OFDM_SYMBOLS_PER_FRAME,
        fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING,
        num_tx=NUM_TX,
        num_streams_per_tx=NUM_STREAMS_PER_TX,
        cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
        num_guard_carriers=NUM_GUARD_CARRIERS,
        dc_null=DC_NULL,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=PILOT_SYMBOLS,
    )


def build_sequence_resource_grid() -> ResourceGrid:
    # This grid is used only to generate one continuous CDL channel
    # over all 80 x 14 OFDM symbols.
    return ResourceGrid(
        num_ofdm_symbols=NUM_FRAMES * OFDM_SYMBOLS_PER_FRAME,
        fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING,
        num_tx=NUM_TX,
        num_streams_per_tx=NUM_STREAMS_PER_TX,
        cyclic_prefix_length=CYCLIC_PREFIX_LENGTH,
        num_guard_carriers=NUM_GUARD_CARRIERS,
        dc_null=DC_NULL,
        pilot_pattern="empty",
    )


def make_channel_model(channel_code: str) -> CDL:
    ut_array = AntennaArray(
        num_rows=1,
        num_cols=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQUENCY_HZ,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
    )

    bs_array = AntennaArray(
        num_rows=1,
        num_cols=2,
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=CARRIER_FREQUENCY_HZ,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
    )

    return CDL(
        model=channel_code,
        delay_spread=DELAY_SPREAD_S,
        carrier_frequency=CARRIER_FREQUENCY_HZ,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="uplink",
        min_speed=UT_SPEED_MPS,
        max_speed=UT_SPEED_MPS,
    )


def derive_cqi(sinr_db: torch.Tensor) -> torch.Tensor:
    # Transparent smoke-test mapping only.
    # Formal experiments will lock and report the full mapping.
    cqi = torch.floor((sinr_db + 6.0) / 2.0)
    return torch.clamp(cqi, min=0.0, max=15.0)


def derive_mcs(cqi: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(cqi)

    target = torch.where(cqi >= 4.0, 1.0, target)
    target = torch.where(cqi >= 8.0, 2.0, target)
    target = torch.where(cqi >= 12.0, 3.0, target)

    # Two-frame normal controller delay.
    applied = torch.empty_like(target)
    applied[:2] = target[0]

    for frame in range(2, target.numel()):
        applied[frame] = target[frame - 2]

    return applied


def channel_correlation(
    estimated_channel: torch.Tensor,
) -> torch.Tensor:
    # Input:
    # [frames, rx, rx_ant, tx, stream, symbol, subcarrier]
    flat = estimated_channel.reshape(NUM_FRAMES, -1)

    output = torch.ones(
        NUM_FRAMES,
        dtype=torch.float32,
        device=estimated_channel.device,
    )

    previous = flat[:-1]
    current = flat[1:]

    numerator = torch.abs(
        torch.sum(torch.conj(previous) * current, dim=1)
    )

    denominator = torch.sqrt(
        torch.sum(torch.abs(previous) ** 2, dim=1)
        * torch.sum(torch.abs(current) ** 2, dim=1)
    )

    output[1:] = numerator / torch.clamp(denominator, min=EPS)

    return torch.clamp(output, min=0.0, max=1.0)


def generate_sequence(
    sample: dict,
) -> tuple[np.ndarray, dict]:
    seed = int(sample["seed"])
    set_all_seeds(seed)

    slot_grid = build_slot_resource_grid()
    sequence_grid = build_sequence_resource_grid()

    stream_manager = StreamManagement(
        np.array([[1]], dtype=np.int32),
        NUM_STREAMS_PER_TX,
    )

    n = int(slot_grid.num_data_symbols * BITS_PER_SYMBOL)
    k = int(n * CODERATE)

    source = BinarySource()

    encoder = LDPC5GEncoder(
        k=k,
        n=n,
        num_bits_per_symbol=BITS_PER_SYMBOL,
    )

    decoder = LDPC5GDecoder(
        encoder=encoder,
        num_iter=DECODER_ITERATIONS,
        hard_out=True,
        return_infobits=True,
    )

    mapper = Mapper(
        constellation_type="qam",
        num_bits_per_symbol=BITS_PER_SYMBOL,
    )

    grid_mapper = ResourceGridMapper(slot_grid)

    estimator = LSChannelEstimator(
        resource_grid=slot_grid,
        interpolation_type="lin",
    )

    equalizer = LMMSEEqualizer(
        slot_grid,
        stream_manager,
    )

    demapper = Demapper(
        demapping_method="app",
        constellation_type="qam",
        num_bits_per_symbol=BITS_PER_SYMBOL,
    )

    remove_nulled = RemoveNulledSubcarriers(slot_grid)

    channel_code = sample["channel_model"].replace("CDL-", "")
    channel_model = make_channel_model(channel_code)

    generate_channel = GenerateOFDMChannel(
        channel_model=channel_model,
        resource_grid=sequence_grid,
        normalize_channel=True,
    )

    apply_channel = ApplyOFDMChannel()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        # One continuous channel realization across all 1,120 symbols.
        continuous_channel = generate_channel(batch_size=1)

        expected_symbols = (
            NUM_FRAMES * OFDM_SYMBOLS_PER_FRAME
        )

        if continuous_channel.shape[-2] != expected_symbols:
            raise RuntimeError(
                "Unexpected continuous channel length: "
                f"{continuous_channel.shape[-2]}"
            )

        # [1, rx, rx_ant, tx, tx_ant, 80, 14, 64]
        channel_reshaped = continuous_channel.reshape(
            1,
            NUM_RX,
            NUM_RX_ANT,
            NUM_TX,
            NUM_STREAMS_PER_TX,
            NUM_FRAMES,
            OFDM_SYMBOLS_PER_FRAME,
            FFT_SIZE,
        )

        # [80, rx, rx_ant, tx, tx_ant, 14, 64]
        channel_frames = (
            channel_reshaped
            .squeeze(0)
            .permute(4, 0, 1, 2, 3, 5, 6)
            .contiguous()
        )

        information_bits = source(
            [
                NUM_FRAMES,
                NUM_TX,
                NUM_STREAMS_PER_TX,
                k,
            ]
        )

        coded_bits = encoder(information_bits)
        data_symbols = mapper(coded_bits)
        transmit_grid = grid_mapper(data_symbols)

        frame_index = torch.arange(
            NUM_FRAMES,
            device=DEVICE,
            dtype=torch.float32,
        )

        # Smooth normal operating profile, 11.5-12.5 dB.
        ebno_db = (
            NORMAL_EBNO_BASE_DB
            + NORMAL_EBNO_SWING_DB
            * torch.sin(
                2.0 * torch.pi * frame_index / 40.0
            )
        )

        noise_variance = ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=BITS_PER_SYMBOL,
            coderate=CODERATE,
            resource_grid=slot_grid,
        )

        noise_broadcast = expand_to_rank(
            noise_variance,
            transmit_grid.ndim,
        )

        received_clean = apply_channel(
            transmit_grid,
            channel_frames,
            None,
        )

        received_grid = apply_channel(
            transmit_grid,
            channel_frames,
            noise_broadcast,
        )

        # ----------------------------------------------------
        # Independent co-channel wideband interferer.
        #
        # The interferer:
        # - is generated with an independent RNG stream;
        # - occupies the same active resource elements;
        # - is present only during [event_start, event_end);
        # - does not modify the desired channel or transmitted bits.
        # ----------------------------------------------------
        event_start = int(sample["event_start"])
        event_end = int(sample["event_end"])

        if not (0 <= event_start < event_end <= NUM_FRAMES):
            raise RuntimeError(
                f"Invalid interference interval: "
                f"[{event_start}, {event_end})"
            )

        interference_generator = torch.Generator(
            device=DEVICE
        )
        interference_generator.manual_seed(
            seed + INTERFERENCE_SEED_OFFSET
        )

        interference_real = torch.randn(
            received_clean.shape,
            device=DEVICE,
            dtype=torch.float32,
            generator=interference_generator,
        )

        interference_imag = torch.randn(
            received_clean.shape,
            device=DEVICE,
            dtype=torch.float32,
            generator=interference_generator,
        )

        raw_interference = (
            interference_real
            + 1j * interference_imag
        ) / torch.sqrt(
            torch.tensor(
                2.0,
                device=DEVICE,
                dtype=torch.float32,
            )
        )

        # Use exactly the active data/pilot resource elements.
        active_mask = (
            torch.abs(
                transmit_grid[:, 0, 0, :, :]
            ) > 0
        ).to(torch.float32)

        active_mask = (
            active_mask[:, None, None, :, :]
            .expand_as(received_clean.real)
        )

        raw_interference = (
            raw_interference * active_mask
        )

        power_dims = (1, 2, 3, 4)

        desired_frame_power = torch.mean(
            torch.abs(received_clean) ** 2,
            dim=power_dims,
        )

        raw_interference_power = torch.mean(
            torch.abs(raw_interference) ** 2,
            dim=power_dims,
        )

        target_interference_power = (
            desired_frame_power
            / (
                10.0
                ** (INTERFERENCE_SIR_DB / 10.0)
            )
        )

        interference_scale = torch.sqrt(
            target_interference_power
            / torch.clamp(
                raw_interference_power,
                min=EPS,
            )
        )

        envelope = torch.zeros(
            NUM_FRAMES,
            device=DEVICE,
            dtype=torch.float32,
        )

        event_length = event_end - event_start
        local_envelope = torch.ones(
            event_length,
            device=DEVICE,
            dtype=torch.float32,
        )

        ramp_length = min(
            INTERFERENCE_RAMP_FRAMES,
            max(1, event_length // 2),
        )

        if ramp_length > 1:
            local_envelope[:ramp_length] = torch.linspace(
                0.35,
                1.0,
                ramp_length,
                device=DEVICE,
            )
            local_envelope[-ramp_length:] = torch.linspace(
                1.0,
                0.35,
                ramp_length,
                device=DEVICE,
            )

        local_index = torch.arange(
            event_length,
            device=DEVICE,
            dtype=torch.float32,
        )

        # Small deterministic temporal fluctuation.
        local_envelope = local_envelope * (
            1.0
            + 0.08
            * torch.sin(
                2.0 * torch.pi * local_index / 7.0
            )
        )

        envelope[event_start:event_end] = local_envelope

        interference_grid = (
            raw_interference
            * interference_scale[
                :, None, None, None, None
            ]
            * envelope[
                :, None, None, None, None
            ]
        )

        received_grid = (
            received_grid + interference_grid
        )

        estimated_channel, estimation_error_variance = estimator(
            received_grid,
            noise_variance,
        )

        # HARMONIZED_FIXED_CSI_AGE_V1
        aged_channel_estimate = estimated_channel.clone()
        aged_estimation_error_variance = (
            estimation_error_variance.clone()
        )

        aged_channel_estimate[
            CSI_AGE_FRAMES:
        ] = estimated_channel[
            :-CSI_AGE_FRAMES
        ]

        aged_estimation_error_variance[
            CSI_AGE_FRAMES:
        ] = estimation_error_variance[
            :-CSI_AGE_FRAMES
        ]

        equalized_symbols, effective_noise_variance = equalizer(
            received_grid,
            aged_channel_estimate,
            aged_estimation_error_variance,
            noise_variance,
        )

        coded_logits = demapper(
            equalized_symbols,
            effective_noise_variance,
        )

        coded_hard = (
            coded_logits > 0
        ).to(coded_bits.dtype)

        decoded_bits = decoder(coded_logits)

        # ----------------------------------------------------
        # Diagnostic control: decode the same received grid
        # using the exact simulated channel instead of LS CSI.
        # This does not enter the KPI sequence or PhyGuard.
        # ----------------------------------------------------
        true_channel_effective = remove_nulled(
            channel_frames
        )

        (
            perfect_equalized_symbols,
            perfect_effective_noise_variance,
        ) = equalizer(
            received_grid,
            true_channel_effective,
            0.0,
            noise_variance,
        )

        perfect_coded_logits = demapper(
            perfect_equalized_symbols,
            perfect_effective_noise_variance,
        )

        perfect_decoded_bits = decoder(
            perfect_coded_logits
        )

        # Per-resource-element 2x2 channel matrix:
        # [frame, symbol, subcarrier, rx_ant, tx_stream]
        channel_matrices = (
            true_channel_effective[
                :, 0, :, 0, :, :, :
            ]
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

        singular_values = torch.linalg.svdvals(
            channel_matrices
        )

        minimum_singular_value = singular_values[..., -1]
        maximum_singular_value = singular_values[..., 0]

        channel_condition_number = (
            maximum_singular_value
            / torch.clamp(
                minimum_singular_value,
                min=1e-8,
            )
        )

        clean_effective = remove_nulled(received_clean)
        received_effective = remove_nulled(received_grid)
        interference_effective = remove_nulled(
            interference_grid
        )

        disturbance_effective = (
            received_effective - clean_effective
        )

        reduce_grid_dims = (1, 2, 3, 4)

        desired_power = torch.mean(
            torch.abs(clean_effective) ** 2,
            dim=reduce_grid_dims,
        )

        total_power = torch.mean(
            torch.abs(received_effective) ** 2,
            dim=reduce_grid_dims,
        )

        measured_noise_power = torch.mean(
            torch.abs(disturbance_effective) ** 2,
            dim=reduce_grid_dims,
        )

        measured_interference_power = torch.mean(
            torch.abs(interference_effective) ** 2,
            dim=reduce_grid_dims,
        )

        achieved_sir_db = db10(
            desired_power
            / torch.clamp(
                measured_interference_power,
                min=EPS,
            )
        )

        mean_event_sir_db = float(
            torch.mean(
                achieved_sir_db[
                    event_start:event_end
                ]
            ).item()
        )

        rsrp_db = db10(desired_power)
        rssi_db = db10(total_power)
        sinr_db = db10(
            desired_power
            / torch.clamp(measured_noise_power, min=EPS)
        )

        evm_rms = torch.sqrt(
            torch.mean(
                torch.abs(equalized_symbols - data_symbols) ** 2,
                dim=(1, 2, 3),
            )
            / torch.clamp(
                torch.mean(
                    torch.abs(data_symbols) ** 2,
                    dim=(1, 2, 3),
                ),
                min=EPS,
            )
        )

        ber_pre_ldpc = torch.mean(
            (coded_hard != coded_bits).to(torch.float32),
            dim=(1, 2, 3),
        )

        block_errors = torch.any(
            decoded_bits != information_bits,
            dim=-1,
        ).to(torch.float32)

        perfect_block_errors = torch.any(
            perfect_decoded_bits != information_bits,
            dim=-1,
        ).to(torch.float32)

        ls_pre_ber_per_stream = (
            (coded_hard != coded_bits)
            .to(torch.float32)
            .mean(dim=(0, 3))
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )

        ls_post_ber_per_stream = (
            (decoded_bits != information_bits)
            .to(torch.float32)
            .mean(dim=(0, 3))
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )

        ls_bler_per_stream = (
            block_errors
            .mean(dim=0)
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )

        perfect_post_ber_per_stream = (
            (
                perfect_decoded_bits
                != information_bits
            )
            .to(torch.float32)
            .mean(dim=(0, 3))
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )

        perfect_csi_bler_per_stream = (
            perfect_block_errors
            .mean(dim=0)
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )

        median_channel_condition_number = float(
            torch.median(
                channel_condition_number
            ).item()
        )

        p95_channel_condition_number = float(
            torch.quantile(
                channel_condition_number,
                0.95,
            ).item()
        )

        median_minimum_singular_value = float(
            torch.median(
                minimum_singular_value
            ).item()
        )

        fraction_condition_number_gt_10 = float(
            torch.mean(
                (
                    channel_condition_number > 10.0
                ).to(torch.float32)
            ).item()
        )

        fraction_condition_number_gt_30 = float(
            torch.mean(
                (
                    channel_condition_number > 30.0
                ).to(torch.float32)
            ).item()
        )

        bler_post_ldpc = torch.mean(
            block_errors,
            dim=(1, 2),
        )

        cqi_index = derive_cqi(sinr_db)
        mcs_index = derive_mcs(cqi_index)

        goodput_ratio = torch.clamp(
            1.0 - bler_post_ldpc,
            min=0.0,
            max=1.0,
        )

        correlation = channel_correlation(
            estimated_channel
        )

        sequence_tensor = torch.stack(
            [
                rsrp_db,
                rssi_db,
                sinr_db,
                evm_rms,
                ber_pre_ldpc,
                bler_post_ldpc,
                cqi_index,
                mcs_index,
                goodput_ratio,
                correlation,
            ],
            dim=1,
        )

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start

    sequence = (
        sequence_tensor
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    runtime = {
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
        ),
        "information_bits_per_stream_per_frame": k,
        "coded_bits_per_stream_per_frame": n,
        "continuous_ofdm_symbols": (
            NUM_FRAMES * OFDM_SYMBOLS_PER_FRAME
        ),
        "intervention_type":
            "independent_cochannel_wideband_interference",
        "event_start": event_start,
        "event_end": event_end,
        "target_interference_sir_db":
            INTERFERENCE_SIR_DB,
        "mean_achieved_event_sir_db":
            mean_event_sir_db,
        "interference_ramp_frames":
            INTERFERENCE_RAMP_FRAMES,
        "ls_pre_ber_per_stream": [
            float(value)
            for value in ls_pre_ber_per_stream
        ],
        "ls_post_ber_per_stream": [
            float(value)
            for value in ls_post_ber_per_stream
        ],
        "ls_bler_per_stream": [
            float(value)
            for value in ls_bler_per_stream
        ],
        "perfect_post_ber_per_stream": [
            float(value)
            for value in perfect_post_ber_per_stream
        ],
        "perfect_csi_bler_per_stream": [
            float(value)
            for value in perfect_csi_bler_per_stream
        ],
        "median_channel_condition_number":
            median_channel_condition_number,
        "p95_channel_condition_number":
            p95_channel_condition_number,
        "median_minimum_singular_value":
            median_minimum_singular_value,
        "fraction_condition_number_gt_10":
            fraction_condition_number_gt_10,
        "fraction_condition_number_gt_30":
            fraction_condition_number_gt_30,
        "receiver_csi_policy":
            "fixed two-frame processing age",
        "receiver_csi_age_frames":
            CSI_AGE_FRAMES,
        "receiver_csi_age_fixed_across_sequence":
            True,
    }

    return sequence, runtime


def validate_sequence(
    sequence: np.ndarray,
    contract: dict,
) -> dict:
    errors = []

    expected_shape = tuple(contract["sequence_shape"])

    if sequence.shape != expected_shape:
        errors.append(
            f"Shape {sequence.shape} != {expected_shape}"
        )

    if sequence.dtype != np.float32:
        errors.append(
            f"dtype {sequence.dtype} != float32"
        )

    if not np.isfinite(sequence).all():
        errors.append("Sequence contains NaN or Inf.")

    range_checks = {}

    for specification in contract["kpis"]:
        index = specification["index"]
        name = specification["name"]
        low, high = specification["valid_range"]

        values = sequence[:, index]

        passed = bool(
            np.all(values >= low)
            and np.all(values <= high)
        )

        range_checks[name] = {
            "pass": passed,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

        if not passed:
            errors.append(
                f"{name} is outside [{low}, {high}]."
            )

        if specification.get("integer_valued", False):
            if not np.allclose(values, np.round(values)):
                errors.append(
                    f"{name} is not integer-valued."
                )

    varying_kpis = int(
        sum(
            np.std(sequence[:, index]) > 1e-7
            for index in range(sequence.shape[1])
        )
    )

    if varying_kpis < 6:
        errors.append(
            f"Only {varying_kpis} KPIs show temporal variation."
        )

    mean_bler = float(np.mean(sequence[:, 5]))
    mean_goodput = float(np.mean(sequence[:, 8]))
    median_correlation = float(np.median(sequence[1:, 9]))

    # Interference samples are evaluated using the
    # event-versus-reference directional checks below.
    # Whole-sequence normal-link limits are not applicable.

    status = "PASS" if not errors else "FAIL"

    return {
        "status": status,
        "shape": list(sequence.shape),
        "dtype": str(sequence.dtype),
        "all_finite": bool(np.isfinite(sequence).all()),
        "varying_kpi_count": varying_kpis,
        "mean_bler": mean_bler,
        "mean_goodput": mean_goodput,
        "median_channel_correlation": median_correlation,
        "range_checks": range_checks,
        "errors": errors,
    }



def validate_interference_direction(
    sequence: np.ndarray,
    sample: dict,
) -> dict:
    # INTERFERENCE_EXCESS_POWER_VALIDATOR_V1
    reference_start = 0
    reference_end = 24

    event_start = int(sample["event_start"])
    event_end = int(sample["event_end"])

    reference = sequence[
        reference_start:reference_end
    ]
    event = sequence[event_start:event_end]

    reference_mean = np.mean(reference, axis=0)
    event_mean = np.mean(event, axis=0)

    rsrp_shift_db = float(
        event_mean[0] - reference_mean[0]
    )
    rssi_increase_db = float(
        event_mean[1] - reference_mean[1]
    )
    sinr_change_db = float(
        event_mean[2] - reference_mean[2]
    )

    rssi_excess_increase_db = float(
        rssi_increase_db - rsrp_shift_db
    )

    reference_evm = float(reference_mean[3])
    event_evm = float(event_mean[3])
    evm_ratio = float(
        event_evm / max(reference_evm, 1e-12)
    )

    reference_ber = float(reference_mean[4])
    event_ber = float(event_mean[4])

    reference_bler = float(reference_mean[5])
    event_bler = float(event_mean[5])

    reference_goodput = float(reference_mean[8])
    event_goodput = float(event_mean[8])

    reference_correlation = float(reference_mean[9])
    event_correlation = float(event_mean[9])

    checks = {
        "desired_rsrp_stable":
            abs(rsrp_shift_db) <= 1.5,
        "rssi_excess_increases":
            rssi_excess_increase_db >= 0.5,
        "sinr_decreases":
            sinr_change_db <= -2.0,
        "evm_increases":
            evm_ratio >= 1.10,
        "pre_ldpc_ber_nonimproving":
            event_ber >= reference_ber,
        "post_ldpc_bler_nonimproving":
            event_bler >= reference_bler,
        "downstream_degradation_present": (
            event_ber > reference_ber + 1e-4
            or event_bler > reference_bler
            or event_goodput < reference_goodput
        ),
    }

    audit_checks = {
        "absolute_rssi_increase_ge_0p5_db":
            rssi_increase_db >= 0.5,
    }

    status = (
        "PASS"
        if all(checks.values())
        else "FAIL"
    )

    return {
        "status": status,
        "reference_interval": [
            reference_start,
            reference_end,
        ],
        "event_interval": [
            event_start,
            event_end,
        ],
        "rsrp_shift_db": rsrp_shift_db,
        "rssi_increase_db": rssi_increase_db,
        "rssi_excess_increase_db":
            rssi_excess_increase_db,
        "sinr_change_db": sinr_change_db,
        "reference_evm_rms": reference_evm,
        "event_evm_rms": event_evm,
        "evm_ratio": evm_ratio,
        "reference_ber_pre_ldpc": reference_ber,
        "event_ber_pre_ldpc": event_ber,
        "reference_bler_post_ldpc": reference_bler,
        "event_bler_post_ldpc": event_bler,
        "reference_goodput": reference_goodput,
        "event_goodput": event_goodput,
        "reference_channel_correlation":
            reference_correlation,
        "event_channel_correlation":
            event_correlation,
        "channel_correlation_change": float(
            event_correlation
            - reference_correlation
        ),
        "checks": checks,
        "audit_checks": audit_checks,
        "validator_policy": {
            "version":
                "interference_excess_power_v1",
            "hard_metric":
                "delta_rssi_minus_delta_rsrp",
            "threshold_db": 0.5,
            "absolute_rssi_check_role":
                "audit_only",
        },
    }

def main() -> None:
    contract = load_json(CONTRACT_PATH)
    plan = load_json(PLAN_PATH)

    sample = select_sample(
        plan,
        os.environ.get("PHYGUARD_SAMPLE_ID", "S24_INTERFERENCE_01"),
    )

    generator_config = {
        "generator_name": "Sionna PHY independent generator",
        "generator_version": LOGICAL_GENERATOR_VERSION,
        "sionna_version": sionna.__version__,
        "torch_version": torch.__version__,
        "device": DEVICE,
        "precision": PRECISION,
        "sample": sample,
        "num_frames": NUM_FRAMES,
        "ofdm_symbols_per_frame": OFDM_SYMBOLS_PER_FRAME,
        "link_topology": "1x2 SIMO",
        "fft_size": FFT_SIZE,
        "active_subcarriers": 52,
        "subcarrier_spacing_hz": SUBCARRIER_SPACING,
        "cyclic_prefix_length": CYCLIC_PREFIX_LENGTH,
        "pilot_symbols": PILOT_SYMBOLS,
        "coding": "5G LDPC",
        "channel_estimator": "LS-linear",
        "equalizer": "LMMSE",
        "ut_speed_mps": UT_SPEED_MPS,
        "delay_spread_seconds": DELAY_SPREAD_S,
        "intervention": {
            "type":
                "independent_cochannel_wideband_interference",
            "event_start": sample["event_start"],
            "event_end": sample["event_end"],
            "target_sir_db": INTERFERENCE_SIR_DB,
            "ramp_frames": INTERFERENCE_RAMP_FRAMES,
            "independent_rng": True,
        },
        "normal_ebno_base_db": NORMAL_EBNO_BASE_DB,
        "normal_ebno_swing_db": NORMAL_EBNO_SWING_DB,
        "receiver_state_policy": {
            "csi_age_frames": CSI_AGE_FRAMES,
            "age_fixed_across_all_classes": True,
            "kpi10_semantics":
                "adjacent estimated-channel correlation",
        },
    }

    config_sha256 = sha256_json(generator_config)

    sequence_first, runtime_first = generate_sequence(sample)
    sequence_second, runtime_second = generate_sequence(sample)

    exact_replay = bool(
        np.array_equal(
            sequence_first,
            sequence_second,
        )
    )

    max_abs_replay_error = float(
        np.max(
            np.abs(
                sequence_first.astype(np.float64)
                - sequence_second.astype(np.float64)
            )
        )
    )

    numerical_replay_pass = (
        max_abs_replay_error <= 1e-6
    )

    validation = validate_sequence(
        sequence_first,
        contract,
    )

    directional_validation = (
        validate_interference_direction(
            sequence_first,
            sample,
        )
    )

    if directional_validation["status"] != "PASS":
        validation["errors"].append(
            "Interference physical-direction "
            "validation failed."
        )
        validation["status"] = "FAIL"

    if not numerical_replay_pass:
        validation["errors"].append(
            "Deterministic numerical replay failed."
        )
        validation["status"] = "FAIL"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    sequence_sha256 = sha256_array(sequence_first)

    np.savez_compressed(
        OUTPUT_DIR / "sequence.npz",
        kpi_sequence=sequence_first,
        kpi_names=np.array(KPI_NAMES),
        frame_index=np.arange(NUM_FRAMES),
    )

    metadata = {
        "sample_id": sample["sample_id"],
        "generator_name": "Sionna PHY independent generator",
        "generator_version": LOGICAL_GENERATOR_VERSION,
        "seed": sample["seed"],
        "label": sample["label"],
        "severity": sample["severity"],
        "channel_model": sample["channel_model"],
        "event_start": sample["event_start"],
        "event_end": sample["event_end"],
        "config_sha256": config_sha256,
        "sequence_sha256": sequence_sha256,
        "kpi_contract_sha256": plan["kpi_contract_sha256"],
        "sequence_shape": list(sequence_first.shape),
        "kpi_names": KPI_NAMES,
        "exact_bitwise_replay": exact_replay,
        "max_abs_replay_error": max_abs_replay_error,
        "numerical_replay_pass": numerical_replay_pass,
        "runtime_first": runtime_first,
        "runtime_second": runtime_second,
        "validation": validation,
        "directional_validation":
            directional_validation,
        "generator_config": generator_config,
    }

    with (
        OUTPUT_DIR / "metadata.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        "status": validation["status"],
        "sample_id": sample["sample_id"],
        "sequence_shape": list(sequence_first.shape),
        "sequence_sha256": sequence_sha256,
        "exact_bitwise_replay": exact_replay,
        "max_abs_replay_error": max_abs_replay_error,
        "numerical_replay_pass": numerical_replay_pass,
        "varying_kpi_count": validation["varying_kpi_count"],
        "normal_ebno_base_db": NORMAL_EBNO_BASE_DB,
        "normal_ebno_swing_db": NORMAL_EBNO_SWING_DB,
        "mean_bler": validation["mean_bler"],
        "mean_goodput": validation["mean_goodput"],
        "median_channel_correlation":
            validation["median_channel_correlation"],
        "elapsed_seconds_first":
            runtime_first["elapsed_seconds"],
        "peak_gpu_memory_mb":
            runtime_first["peak_gpu_memory_mb"],
        "ls_pre_ber_per_stream":
            runtime_first["ls_pre_ber_per_stream"],
        "ls_post_ber_per_stream":
            runtime_first["ls_post_ber_per_stream"],
        "ls_bler_per_stream":
            runtime_first["ls_bler_per_stream"],
        "perfect_post_ber_per_stream":
            runtime_first[
                "perfect_post_ber_per_stream"
            ],
        "perfect_csi_bler_per_stream":
            runtime_first[
                "perfect_csi_bler_per_stream"
            ],
        "median_channel_condition_number":
            runtime_first[
                "median_channel_condition_number"
            ],
        "p95_channel_condition_number":
            runtime_first[
                "p95_channel_condition_number"
            ],
        "median_minimum_singular_value":
            runtime_first[
                "median_minimum_singular_value"
            ],
        "fraction_condition_number_gt_10":
            runtime_first[
                "fraction_condition_number_gt_10"
            ],
        "fraction_condition_number_gt_30":
            runtime_first[
                "fraction_condition_number_gt_30"
            ],
        "target_interference_sir_db":
            INTERFERENCE_SIR_DB,
        "mean_achieved_event_sir_db":
            runtime_first[
                "mean_achieved_event_sir_db"
            ],
        "directional_validation":
            directional_validation,
        "output_directory": str(OUTPUT_DIR),
        "errors": validation["errors"],
    }

    with (
        RESULT_DIR / "report.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))

    if validation["status"] != "PASS":
        raise RuntimeError(
            "Interference 80-frame sequence validation failed."
        )


if __name__ == "__main__":
    main()
