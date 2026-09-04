MECHANISM: Four-step Muon orthogonalization for higher token throughput

HYPOTHESIS: Using the best 258,048-token batch with four rather than five Muon polar iterations will process more than 522M tokens and lower val_bpb below 0.985181.

INTENDED_EDIT: Restore the best-performing 126-sequence microbatch and remove the final Muon orthogonalization iteration.

EVIDENCE: The 126-sequence design achieved the best observed val_bpb, 0.985181, while processing 522.0M tokens; reducing optimizer matrix work tests whether additional throughput improves that result without materially weakening Muon updates.

<<<<<<< SEARCH
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=4, beta2=0.95, weight_decay=weight_decay,
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 31 * 2**13 # ~254K tokens; one 124-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 124  # one microbatch per 253,952-token optimizer step
=======
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
>>>>>>> REPLACE