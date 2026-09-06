# Ten-digit addition continuation on T4

The operator authorized only B03-C3 (65 proposals) and B04-C2 (64 proposals)
in `controlled-openevolve-transformer-v2-1-mps-campaign` to continue to 100.
C4 and the other eighteen C0-C3 trajectories are excluded.

`modal_addition_t4_app.py` deploys `rl4rl-ten-digit-addition-t4` with one T4,
four CPU cores, and 16 GiB memory per evaluation, at most two containers.
It uses the shared evaluator and protected AdderBoard verifier. The task
amendment changes training device from MPS to CUDA and backend to Modal;
training steps, qualification rules and the 1,800-second evaluation timeout
remain unchanged. GPU results need not be numerically identical to MPS.

The bounded launcher is `python -m experiments.resume_trajectory_to_target`
with `--target 100` and `C0C3_MODAL_APP=rl4rl-ten-digit-addition-t4`.
The synchronous pause control prevents starting proposal 101. This launch
does not configure CPU fallback.

The campaign's `amendments/t4-to100-20260906` directory contains original
manifests, task/protocol, selected run states and event histories, hashes of
untouched runs, the original-seed T4 verification, and launch receipts.
These local campaign artifacts remain gitignored. Recovery should retrieve
existing results from the dedicated Modal result dictionary before charging
an interrupted proposal; never blindly replay an evaluation.

The original seed probe completed training and protected verification on Tesla
T4 in 180.65 worker seconds, but qualified only 7,434/10,010 cases (74.27%).
It is recorded as nonqualification, not a passing calibration. CUDA continuation
retains the 99% gate and original historical seed/incumbent fitness; this probe
is evidence of backend sensitivity, and must accompany comparisons with MPS.
