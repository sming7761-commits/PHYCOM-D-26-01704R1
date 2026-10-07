# Paired uncertainty analysis

All 14 baselines and PhyGuard were compared on exactly aligned repeat/index/truth records.

PhyGuard means: selected accuracy 0.9432, coverage 0.8319, false-specific rate 0.0556, physical exact match 0.7847, and control abstention 0.9444.

## Models with a point selected-accuracy gain

- MINIROCKET: selected-accuracy delta +0.0101 [-0.0177, +0.0379]; false-specific-rate delta +0.0778 [+0.0268, +0.1312]; control-abstention delta -0.0778 [-0.1312, -0.0268].
- MULTIROCKET: selected-accuracy delta +0.0184 [-0.0053, +0.0432]; false-specific-rate delta +0.0722 [+0.0251, +0.1219]; control-abstention delta -0.0722 [-0.1219, -0.0251].
- MULTIROCKET_HYDRA: selected-accuracy delta +0.0215 [-0.0030, +0.0465]; false-specific-rate delta +0.0806 [+0.0307, +0.1332]; control-abstention delta -0.0806 [-0.1332, -0.0307].
- TCN: selected-accuracy delta +0.0015 [-0.0275, +0.0303]; false-specific-rate delta +0.0833 [+0.0369, +0.1305]; control-abstention delta -0.0833 [-0.1305, -0.0369].

## Reporting boundary

No baseline may be described as uniformly superior unless it improves selected accuracy and coverage while not worsening false-specific rate, physical exact match, or control abstention.
