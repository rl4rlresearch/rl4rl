# TinyKWS-RNN task

TinyKWS-RNN is the protocol-2.1 source-only Mini Speech Commands stratum. It
classifies eight commands from protected 32×20 log-mel sequences with a genuine
causal recurrent model and minimizes exact executed dense inference MACs after
meeting a speaker-disjoint validation-accuracy gate.

## Frozen task boundary

- Dataset: checksummed TensorFlow Mini Speech Commands archive.
- Labels: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, and `yes`.
- Split: deterministic speaker hashes create disjoint training, public
  validation, and sealed Layer-C speaker populations.
- Frontend: protected 16 kHz waveform normalization, 512-point STFT,
  400-sample window, 160-sample hop, 20 mel bands, log compression,
  interpolation to 32 frames, and training-split per-band normalization.
- Exposure: exactly 50,000 training examples per candidate.
- Editable surface: only `train.py`.
- Backend: local CPU. MPS calibration was rejected because recurrent sequence
  execution was substantially slower on this host.
- Parameter ceiling: 100,000 learned parameters.
- Qualification: at least 85% public validation accuracy. The originally
  proposed 90% threshold was reduced during prospective calibration after
  recurrent seeds below 100,000 parameters reached 85–87.5% under the fixed
  50,000-example exposure.

The starting source is a one-layer, 128-unit causal GRU with an online mean
summary. It has 58,672 learned parameters and exposes both a per-frame step and
an equivalent sequence-kernel fast path.

## Objective and recurrence checks

The evaluator hooks every allowed learned matrix module and counts its actually
executed dense MACs. It rejects direct or functional matrix multiplication,
bidirectional recurrence, uncounted parameter-bearing modules, and dense
zero-weight tricks. It records total MACs across the complete validation set,
total recurrent steps, parameter count, hidden-state size, accuracy,
cross-entropy, exit-depth summaries, exposure, and timing.

The exact integer objective is lexicographic:

```text
inference_cost =
    (total_inference_macs * (validation_cases * 64 + 1)
     + recurrent_steps)
    * (100000 + 1)
    + parameters
```

Thus one MAC always dominates every possible step and parameter tie-break; one
step dominates every possible parameter tie-break. The controller preserves
the integer without converting it through a floating-point score.

The model must expose `initial_state`, `recurrent_step`, and `classify`.
Optional `frame_schedule`, `recurrent_sequence`, and `exit_mask` methods permit
causal striding, efficient standard recurrent kernels, and dynamic early exits.
The evaluator checks that state changes, later state depends on earlier state,
logits depend on recurrent output, recurrent-path weights train, a sequence
fast path matches repeated causal steps, and every executed step is counted.

## Campaign preset

The supplied v2.1 preset declares five blocks and C0–C4 in each block, 200
physical proposals per trajectory, GPT-5.6 Sol xhigh, and assumption-changing
prompts every tenth proposal for C1/C3. C4 is the separately reported periodic
full-refresh control. Its normal supervisor advances only C0–C3; the C4 profile
is separate so those five trajectories can remain dormant.

TinyKWS evaluators use a dedicated ten-slot host lock namespace rather than the
shared AdderBoard local-evaluator namespace. This is scheduler isolation; all
processes still run on the same physical Mac and can share CPU, memory, and
thermal limits.

## Commands

From the repository root:

```bash
PY=architecture_discovery/.venv/bin/python
CLI=experiments.c0c3_factorial.cli

$PY -m experiments.c0c3_factorial.tiny_kws_rnn prepare --repo-root .

$PY -m $CLI calibrate \
  --protocol experiments/c0c3_factorial/configs/protocols/tiny_kws_rnn_openevolve_v2_1.toml \
  --task experiments/c0c3_factorial/configs/tasks/tiny_kws_rnn_source_only_cpu.toml \
  --output data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-calibration \
  --python-bin "$PY"

$PY -m $CLI create \
  --protocol experiments/c0c3_factorial/configs/protocols/tiny_kws_rnn_openevolve_v2_1.toml \
  --task experiments/c0c3_factorial/configs/tasks/tiny_kws_rnn_source_only_cpu.toml \
  --framework experiments/c0c3_factorial/configs/frameworks/openevolve_tiny_kws_rnn_v2_1.toml \
  --baseline data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-calibration/baseline.json \
  --output data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign
```

The durable C0–C3 profile is `openevolve-v2.1-tiny-kws-rnn`; the separate C4
profile is `openevolve-v2.1-tiny-kws-rnn-c4`.
