MECHANISM: Staged early local-to-global context funnel

HYPOTHESIS: Using a 256-token window in the layer preceding the existing 512-token first pre-global layer will achieve `val_bpb < 0.982068` by improving early contextual aggregation at modest compute cost.

INTENDED_EDIT: Retain the 512-token first pre-global window and 128-token default local windows, but expand the immediately preceding local layer to 256 tokens.

EVIDENCE: Expanding only the first pre-global window from 256 to 512 tokens improved `val_bpb` from `0.982234` to `0.982068`, whereas expanding it to 1,024 regressed to `0.983064`; this motivates distributing additional early context into the preceding layer instead of further enlarging the boundary window.

<<<<<<< SEARCH
            else:
                next_char = pattern[(layer_idx + 1) % len(pattern)]
                is_first_pre_global = next_char == "L" and layer_idx < config.n_layer // 2
                window = pre_global_window if is_first_pre_global else short_window
=======
            else:
                next_char = pattern[(layer_idx + 1) % len(pattern)]
                next_next_char = pattern[(layer_idx + 2) % len(pattern)]
                is_first_pre_global = next_char == "L" and layer_idx < config.n_layer // 2
                is_early_pre_global_feeder = next_next_char == "L" and layer_idx < config.n_layer // 2
                if is_first_pre_global:
                    window = pre_global_window
                elif is_early_pre_global_feeder:
                    window = long_window // 8
                else:
                    window = short_window
>>>>>>> REPLACE