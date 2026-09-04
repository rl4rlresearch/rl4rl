MECHANISM: Stratified antithetic crop scheduling

HYPOTHESIS: Deterministically balancing all 25 full-crop offsets and nine central offsets will exceed 9,239 correct predictions by reducing augmentation-sampling variance without the runtime cost of per-example crop selection.

INTENDED_EDIT: Replace randomly sampled batch-shared crop offsets with inexpensive step-indexed permutations that cover each offset nearly uniformly across 782 optimizer steps.

EVIDENCE: Per-example crop sampling targeted augmentation diversity but timed out; the best cosine-curriculum design still uses randomly imbalanced batch-shared offsets, so deterministic stratification tests the same subject area without unfolding or additional model evaluations.

<<<<<<< SEARCH
    offsets = torch.randint(0, 5, (2,))
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    central_offsets = torch.randint(1, 4, (2,))
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
=======
    full_offset_index = (7 * step + 3) % 25
    offset_y = full_offset_index // 5
    offset_x = full_offset_index % 5
    central_offset_index = (5 * step + 1) % 9
    central_y = 1 + central_offset_index // 3
    central_x = 1 + central_offset_index % 3
>>>>>>> REPLACE