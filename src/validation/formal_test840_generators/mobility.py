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
LOGICAL_GENERATOR_VERSION = "sionna_simo_mobility_csi_aging_v1.0.0"

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

# Locked mild mobility intervention.
# Baseline: 3 m/s (10.8 km/h)
# Event peak: 18 m/s (64.8 km/h)
MOBILITY_EVENT_SPEED_MPS = float(
    os.environ.get("MOBILITY_EVENT_SPEED_MPS", "18.0")
)
MOBILITY_RAMP_FRAMES = 4

# MOBILITY_CSI_AGING_FIX_V1
# Fixed receiver processing latency, applied to all frames.
CSI_AGE_FRAMES = 2
# Baseline and high-mobility channels must share the same
# random CDL realization. Only the velocity parameter changes.
MOBILITY_CHANNEL_SEED_OFFSET = 0
MOBILITY_DATA_SEED_OFFSET = 3000007

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


def make_channel_model(
    channel_code: str,
    speed_mps: float,
) -> CDL:
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
        min_speed=speed_mps,
        max_speed=speed_mps,
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



def csi_freshness_correlation(
    current_channel: torch.Tensor,
    receiver_visible_channel: torch.Tensor,
) -> torch.Tensor:
    current_flat = current_channel.reshape(
        NUM_FRAMES,
        -1,
    )

    visible_flat = receiver_visible_channel.reshape(
        NUM_FRAMES,
        -1,
    )

    numerator = torch.abs(
        torch.sum(
            torch.conj(visible_flat) * current_flat,
            dim=1,
        )
    )

    denominator = torch.sqrt(
        torch.sum(
            torch.abs(visible_flat) ** 2,
            dim=1,
        )
        * torch.sum(
            torch.abs(current_flat) ** 2,
            dim=1,
        )
    )

    correlation = numerator / torch.clamp(
        denominator,
        min=EPS,
    )

    return torch.clamp(
        correlation,
        min=0.0,
        max=1.0,
    )


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

    baseline_channel_model = make_channel_model(
        channel_code,
        UT_SPEED_MPS,
    )

    mobility_channel_model = make_channel_model(
        channel_code,
        MOBILITY_EVENT_SPEED_MPS,
    )

    generate_baseline_channel = GenerateOFDMChannel(
        channel_model=baseline_channel_model,
        resource_grid=sequence_grid,
        normalize_channel=True,
    )

    generate_mobility_channel = GenerateOFDMChannel(
        channel_model=mobility_channel_model,
        resource_grid=sequence_grid,
        normalize_channel=True,
    )

    apply_channel = ApplyOFDMChannel()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
                # ----------------------------------------------------
        # Generate matched continuous CDL trajectories.
        # Both calls use the same random realization; the controlled
        # difference is the user-terminal velocity only.
        # ----------------------------------------------------
        set_all_seeds(seed)

        baseline_continuous_channel = (
            generate_baseline_channel(batch_size=1)
        )

        set_all_seeds(
            seed + MOBILITY_CHANNEL_SEED_OFFSET
        )

        mobility_continuous_channel = (
            generate_mobility_channel(batch_size=1)
        )

        expected_symbols = (
            NUM_FRAMES * OFDM_SYMBOLS_PER_FRAME
        )

        for name, tensor in (
            (
                "baseline",
                baseline_continuous_channel,
            ),
            (
                "mobility",
                mobility_continuous_channel,
            ),
        ):
            if tensor.shape[-2] != expected_symbols:
                raise RuntimeError(
                    f"Unexpected {name} channel length: "
                    f"{tensor.shape[-2]}"
                )

        def reshape_channel(
            continuous: torch.Tensor,
        ) -> torch.Tensor:
            reshaped = continuous.reshape(
                1,
                NUM_RX,
                NUM_RX_ANT,
                NUM_TX,
                NUM_STREAMS_PER_TX,
                NUM_FRAMES,
                OFDM_SYMBOLS_PER_FRAME,
                FFT_SIZE,
            )

            return (
                reshaped
                .squeeze(0)
                .permute(4, 0, 1, 2, 3, 5, 6)
                .contiguous()
            )

        baseline_channel_frames = reshape_channel(
            baseline_continuous_channel
        )

        mobility_channel_frames = reshape_channel(
            mobility_continuous_channel
        )

        event_start = int(sample["event_start"])
        event_end = int(sample["event_end"])

        if not (
            0 <= event_start < event_end <= NUM_FRAMES
        ):
            raise RuntimeError(
                f"Invalid mobility interval: "
                f"[{event_start}, {event_end})"
            )

        # Match frame-wise channel power before blending so that
        # mobility is represented primarily by temporal variation,
        # rather than by an unintended path-loss intervention.
        channel_reduce_dims = (1, 2, 3, 4, 5, 6)

        baseline_frame_power = torch.mean(
            torch.abs(
                baseline_channel_frames
            ) ** 2,
            dim=channel_reduce_dims,
        )

        mobility_frame_power = torch.mean(
            torch.abs(
                mobility_channel_frames
            ) ** 2,
            dim=channel_reduce_dims,
        )

        mobility_channel_frames = (
            mobility_channel_frames
            * torch.sqrt(
                baseline_frame_power
                / torch.clamp(
                    mobility_frame_power,
                    min=EPS,
                )
            )[
                :, None, None, None,
                None, None, None
            ]
        )

        mobility_envelope = torch.zeros(
            NUM_FRAMES,
            dtype=torch.float32,
            device=DEVICE,
        )

        event_length = event_end - event_start

        local_envelope = torch.ones(
            event_length,
            dtype=torch.float32,
            device=DEVICE,
        )

        ramp_length = min(
            MOBILITY_RAMP_FRAMES,
            max(1, event_length // 2),
        )

        if ramp_length > 1:
            local_envelope[:ramp_length] = (
                torch.linspace(
                    0.25,
                    1.0,
                    ramp_length,
                    device=DEVICE,
                )
            )

            local_envelope[-ramp_length:] = (
                torch.linspace(
                    1.0,
                    0.25,
                    ramp_length,
                    device=DEVICE,
                )
            )

        mobility_envelope[
            event_start:event_end
        ] = local_envelope

        blend_weight = mobility_envelope[
            :, None, None, None,
            None, None, None
        ]

        blended_channel_frames = (
            baseline_channel_frames
            * (1.0 - blend_weight)
            + mobility_channel_frames
            * blend_weight
        )

        # Remove accidental power changes caused by complex
        # cross-fading between the two trajectories.
        blended_frame_power = torch.mean(
            torch.abs(
                blended_channel_frames
            ) ** 2,
            dim=channel_reduce_dims,
        )

        channel_frames = (
            blended_channel_frames
            * torch.sqrt(
                baseline_frame_power
                / torch.clamp(
                    blended_frame_power,
                    min=EPS,
                )
            )[
                :, None, None, None,
                None, None, None
            ]
        )

        applied_speed_mps = (
            UT_SPEED_MPS
            + mobility_envelope
            * (
                MOBILITY_EVENT_SPEED_MPS
                - UT_SPEED_MPS
            )
        )

        mean_applied_event_speed_mps = float(
            torch.mean(
                applied_speed_mps[
                    event_start:event_end
                ]
            ).item()
        )

        maximum_applied_speed_mps = float(
            torch.max(
                applied_speed_mps[
                    event_start:event_end
                ]
            ).item()
        )

        # A separate deterministic stream is used for payload bits
        # and receiver noise.
        set_all_seeds(
            seed + MOBILITY_DATA_SEED_OFFSET
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

        # MOBILITY_EFFECTIVE_POWER_FIX_V3
        # ----------------------------------------------------
        # Mobility counterfactual invariant:
        #
        # During the event, the desired received power produced
        # by H*x is matched to the reference-window mean.
        # This prevents a mobility intervention from becoming
        # an unintended blockage intervention.
        #
        # The positive frame-wise scalar does not change the
        # normalized temporal channel-correlation direction.
        # ----------------------------------------------------
        baseline_received_for_calibration = apply_channel(
            transmit_grid,
            baseline_channel_frames,
            None,
        )

        mobility_received_before_calibration = apply_channel(
            transmit_grid,
            channel_frames,
            None,
        )

        effective_power_dims = (1, 2, 3, 4)

        baseline_effective_power = torch.mean(
            torch.abs(
                baseline_received_for_calibration
            ) ** 2,
            dim=effective_power_dims,
        )

        mobility_effective_power_before = torch.mean(
            torch.abs(
                mobility_received_before_calibration
            ) ** 2,
            dim=effective_power_dims,
        )

        reference_power_target = torch.mean(
            baseline_effective_power[0:24]
        )

        event_power_scale = torch.sqrt(
            reference_power_target
            / torch.clamp(
                mobility_effective_power_before,
                min=EPS,
            )
        )

        applied_power_scale = torch.ones(
            NUM_FRAMES,
            dtype=torch.float32,
            device=DEVICE,
        )

        applied_power_scale[
            event_start:event_end
        ] = event_power_scale[
            event_start:event_end
        ]

        channel_frames = (
            channel_frames
            * applied_power_scale[
                :, None, None, None,
                None, None, None
            ]
        )

        calibrated_received_clean = apply_channel(
            transmit_grid,
            channel_frames,
            None,
        )

        calibrated_effective_power = torch.mean(
            torch.abs(
                calibrated_received_clean
            ) ** 2,
            dim=effective_power_dims,
        )

        event_effective_power_error_db = (
            10.0
            * torch.log10(
                torch.clamp(
                    calibrated_effective_power[
                        event_start:event_end
                    ],
                    min=EPS,
                )
                / torch.clamp(
                    reference_power_target,
                    min=EPS,
                )
            )
        )

        mean_abs_effective_rsrp_error_db = float(
            torch.mean(
                torch.abs(
                    event_effective_power_error_db
                )
            ).item()
        )

        max_abs_effective_rsrp_error_db = float(
            torch.max(
                torch.abs(
                    event_effective_power_error_db
                )
            ).item()
        )

        mean_event_power_scale = float(
            torch.mean(
                applied_power_scale[
                    event_start:event_end
                ]
            ).item()
        )

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

        received_clean = calibrated_received_clean

        received_grid = apply_channel(
            transmit_grid,
            channel_frames,
            noise_broadcast,
        )

        estimated_channel, estimation_error_variance = estimator(
            received_grid,
            noise_variance,
        )


        if estimated_channel.shape[0] != NUM_FRAMES:
            raise RuntimeError(
                "Unexpected CSI frame dimension."
            )

        receiver_channel_estimate = (
            estimated_channel.clone()
        )

        receiver_estimation_error_variance = (
            estimation_error_variance.clone()
        )

        receiver_channel_estimate[
            CSI_AGE_FRAMES:
        ] = estimated_channel[
            :-CSI_AGE_FRAMES
        ]

        receiver_estimation_error_variance[
            CSI_AGE_FRAMES:
        ] = estimation_error_variance[
            :-CSI_AGE_FRAMES
        ]

        equalized_symbols, effective_noise_variance = equalizer(
            received_grid,
            receiver_channel_estimate,
            receiver_estimation_error_variance,
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
        noise_effective = received_effective - clean_effective

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
            torch.abs(noise_effective) ** 2,
            dim=reduce_grid_dims,
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
            "time_localized_mobility_increase",
        "event_start": event_start,
        "event_end": event_end,
        "baseline_speed_mps": UT_SPEED_MPS,
        "event_target_speed_mps":
            MOBILITY_EVENT_SPEED_MPS,
        "mean_applied_event_speed_mps":
            mean_applied_event_speed_mps,
        "maximum_applied_speed_mps":
            maximum_applied_speed_mps,
        "mobility_ramp_frames":
            MOBILITY_RAMP_FRAMES,
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
        "effective_received_power_control":
            "event matched to reference-window mean",
        "mean_abs_effective_rsrp_error_db":
            mean_abs_effective_rsrp_error_db,
        "max_abs_effective_rsrp_error_db":
            max_abs_effective_rsrp_error_db,
        "mean_event_power_scale":
            mean_event_power_scale,
        "receiver_csi_policy":
            "fixed-age CSI for every frame",
        "receiver_csi_age_frames":
            CSI_AGE_FRAMES,
        "receiver_csi_age_fixed_across_sequence":
            True,
        "mobility_changed_factor":
            "user-terminal velocity only",
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

    # Mobility samples are checked using mechanism-specific
    # directional constraints instead of normal health limits.

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



def validate_mobility_direction(
    sequence: np.ndarray,
    sample: dict,
) -> dict:
    reference_start = 0
    reference_end = 24

    event_start = int(sample["event_start"])
    event_end = int(sample["event_end"])

    reference = sequence[
        reference_start:reference_end
    ]

    event = sequence[
        event_start:event_end
    ]

    reference_mean = np.mean(
        reference,
        axis=0,
    )

    event_mean = np.mean(
        event,
        axis=0,
    )

    rsrp_change_db = float(
        event_mean[0] - reference_mean[0]
    )

    rssi_change_db = float(
        event_mean[1] - reference_mean[1]
    )

    sinr_change_db = float(
        event_mean[2] - reference_mean[2]
    )

    reference_evm = float(
        reference_mean[3]
    )

    event_evm = float(
        event_mean[3]
    )

    evm_ratio = float(
        event_evm
        / max(reference_evm, 1e-12)
    )

    reference_ber = float(
        reference_mean[4]
    )

    event_ber = float(
        event_mean[4]
    )

    reference_bler = float(
        reference_mean[5]
    )

    event_bler = float(
        event_mean[5]
    )

    reference_goodput = float(
        reference_mean[8]
    )

    event_goodput = float(
        event_mean[8]
    )

    reference_correlation = float(
        reference_mean[9]
    )

    event_correlation = float(
        event_mean[9]
    )

    correlation_change = float(
        event_correlation
        - reference_correlation
    )

    receiver_stress_present = bool(
        evm_ratio >= 1.05
        or event_ber
        >= reference_ber + 1e-4
        or event_bler > reference_bler
    )

    checks = {
        # Power is normalized deliberately so mobility remains
        # distinct from blockage and interference.
        "rsrp_approximately_stable":
            abs(rsrp_change_db) <= 1.5,

        "rssi_approximately_stable":
            abs(rssi_change_db) <= 2.0,

        "sinr_not_dominated_by_power_loss":
            abs(sinr_change_db) <= 2.5,

        "channel_correlation_not_strongly_improving":
            correlation_change <= 0.02,

        "event_channel_correlation_bounded":
            event_correlation <= 0.98,

        "receiver_stress_present":
            receiver_stress_present,

        "goodput_nonimproving":
            event_goodput
            <= reference_goodput + 1e-7,
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
        "rsrp_change_db":
            rsrp_change_db,
        "rssi_change_db":
            rssi_change_db,
        "sinr_change_db":
            sinr_change_db,
        "reference_evm_rms":
            reference_evm,
        "event_evm_rms":
            event_evm,
        "evm_ratio":
            evm_ratio,
        "reference_ber_pre_ldpc":
            reference_ber,
        "event_ber_pre_ldpc":
            event_ber,
        "reference_bler_post_ldpc":
            reference_bler,
        "event_bler_post_ldpc":
            event_bler,
        "reference_goodput":
            reference_goodput,
        "event_goodput":
            event_goodput,
        "reference_channel_correlation":
            reference_correlation,
        "event_channel_correlation":
            event_correlation,
        "channel_correlation_change":
            correlation_change,
        "receiver_stress_present":
            receiver_stress_present,
        "checks": checks,
    }

def main() -> None:
    contract = load_json(CONTRACT_PATH)
    plan = load_json(PLAN_PATH)

    sample = select_sample(
        plan,
        os.environ.get("PHYGUARD_SAMPLE_ID", "S24_MOBILITY_01"),
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
        "baseline_ut_speed_mps": UT_SPEED_MPS,
        "intervention": {
            "type":
                "time_localized_mobility_increase",
            "event_start": sample["event_start"],
            "event_end": sample["event_end"],
            "event_target_speed_mps":
                MOBILITY_EVENT_SPEED_MPS,
            "ramp_frames":
                MOBILITY_RAMP_FRAMES,
            "frame_power_normalized":
                True,
            "matched_cdl_random_realization":
                True,
            "single_factor_intervention":
                "user_terminal_velocity",
        },
        "delay_spread_seconds": DELAY_SPREAD_S,
        "normal_ebno_base_db": NORMAL_EBNO_BASE_DB,
        "normal_ebno_swing_db": NORMAL_EBNO_SWING_DB,
        "mobility_counterfactual_control": {
            "single_changed_factor":
                "user_terminal_velocity",
            "effective_received_power_control":
                "event matched to reference-window mean",
            "reference_window": [0, 24],
            "event_window_from_sample": True,
        },
        "receiver_state_policy": {
            "csi_age_frames": CSI_AGE_FRAMES,
            "age_fixed_across_normal_and_event": True,
            "warmup_uses_current_csi": True,
            "mobility_changed_factor":
                "user-terminal velocity only",
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
        validate_mobility_direction(
            sequence_first,
            sample,
        )
    )

    if directional_validation["status"] != "PASS":
        validation["errors"].append(
            "Mobility physical-direction "
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
        "baseline_speed_mps":
            UT_SPEED_MPS,
        "event_target_speed_mps":
            MOBILITY_EVENT_SPEED_MPS,
        "mean_applied_event_speed_mps":
            runtime_first[
                "mean_applied_event_speed_mps"
            ],
        "maximum_applied_speed_mps":
            runtime_first[
                "maximum_applied_speed_mps"
            ],
        "directional_validation":
            directional_validation,
        "output_directory": str(OUTPUT_DIR),
        "errors": validation["errors"],
        "effective_received_power_control":
            runtime_first[
                "effective_received_power_control"
            ],
        "mean_abs_effective_rsrp_error_db":
            runtime_first[
                "mean_abs_effective_rsrp_error_db"
            ],
        "max_abs_effective_rsrp_error_db":
            runtime_first[
                "max_abs_effective_rsrp_error_db"
            ],
        "mean_event_power_scale":
            runtime_first[
                "mean_event_power_scale"
            ],
        "receiver_csi_policy":
            runtime_first[
                "receiver_csi_policy"
            ],
        "receiver_csi_age_frames":
            runtime_first[
                "receiver_csi_age_frames"
            ],
        "receiver_csi_age_fixed_across_sequence":
            runtime_first[
                "receiver_csi_age_fixed_across_sequence"
            ],
    }

    with (
        RESULT_DIR / "report.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))

    if validation["status"] != "PASS":
        raise RuntimeError(
            "Mobility 80-frame sequence validation failed."
        )


if __name__ == "__main__":
    main()
