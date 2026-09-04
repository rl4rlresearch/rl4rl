MECHANISM: Four-step Polar Express Muon orthogonalization

HYPOTHESIS: Reducing Muon orthogonalization from five iterations to four will preserve update quality while lowering optimizer overhead enough to exceed 489.9M training tokens and improve val_bpb below 0.987174.

INTENDED_EDIT: Use four Newton–Schulz/Polar Express iterations for every Muon parameter group, leaving the winning model, batch size, learning rates, and schedules unchanged.

EVIDENCE: The 262K-token design is best at 0.987174, while subsequent batch, attention, learning-rate, schedule, and optimizer-statistics changes failed to improve it; reducing one orthogonalization iteration isolates a throughput optimization without altering those validated choices.

<<<<<<< SEARCH
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=4, beta2=0.95, weight_decay=weight_decay,
>>>>>>> REPLACE