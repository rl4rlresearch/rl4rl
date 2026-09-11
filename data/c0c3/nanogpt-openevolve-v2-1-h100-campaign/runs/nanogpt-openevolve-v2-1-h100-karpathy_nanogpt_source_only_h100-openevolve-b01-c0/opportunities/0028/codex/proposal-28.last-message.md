MECHANISM: Max-autotuned graph compilation

HYPOTHESIS: Enabling maximum kernel autotuning will beat 0.984313 val_bpb by increasing steady-state throughput beyond 506.2M tokens while preserving the proven model and optimization settings.

INTENDED_EDIT: Compile the model in PyTorch’s max-autotune mode; compilation remains outside the measured training window.

EVIDENCE: The best 78.0% run processed 506.2M tokens, whereas adjacent 78.1% and 77.9% runs processed only 500.2M and 500.7M tokens and regressed to 0.985030 and 0.984883, motivating a throughput-only change.

<<<<<<< SEARCH
model = torch.compile(model, dynamic=False)
=======
model = torch.compile(model, dynamic=False, mode="max-autotune")
>>>>>>> REPLACE