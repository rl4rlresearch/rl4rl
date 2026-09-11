MECHANISM: Front-loaded local-to-global context allocation

HYPOTHESIS: Retaining a 256-token window only before the first full-context layer will achieve `val_bpb < 0.982545` by preserving the longer-lived early aggregation benefit while recovering some throughput from the second boundary.

INTENDED_EDIT: Use 256-token attention at the first pre-global layer and 128-token attention at all other local layers; retain both full-context layers.

EVIDENCE: Using 256-token windows at both pre-global boundaries improved `val_bpb` from `0.982806` to `0.982545`, while expanding both to 512 tokens regressed to `0.982674`; isolating the early boundary tests whether the gain is concentrated where contextualized representations propagate through more downstream layers.

<<<<<<< SEARCH
            else:
                next_char = pattern[(layer_idx + 1) % len(pattern)]
                window = pre_global_window if next_char == "L" else short_window
=======
            else:
                next_char = pattern[(layer_idx + 1) % len(pattern)]
                is_first_pre_global = next_char == "L" and layer_idx < config.n_layer // 2
                window = pre_global_window if is_first_pre_global else short_window
>>>>>>> REPLACE