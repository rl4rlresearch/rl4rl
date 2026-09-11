MECHANISM: Single-microbatch optimizer steps

HYPOTHESIS: Doubling the device batch to 256 will fit within H100 memory and reduce per-step launch and gradient-accumulation overhead, increasing trained tokens enough to lower val_bpb below 0.993365.

INTENDED_EDIT: Process the full 524,288-token optimizer batch in one forward/backward pass instead of two accumulated microbatches.

EVIDENCE: The best design peaks at 45,060 MB VRAM while requiring two microbatches per optimizer step, leaving device-batch scaling as an untested throughput lever without changing the successful model or attention topology.

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 256  # full optimizer batch in one microstep
>>>>>>> REPLACE