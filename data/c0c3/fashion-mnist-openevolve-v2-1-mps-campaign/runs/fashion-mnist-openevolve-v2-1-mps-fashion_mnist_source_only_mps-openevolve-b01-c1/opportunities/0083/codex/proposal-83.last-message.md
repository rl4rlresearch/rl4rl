MECHANISM: Low-discrepancy batch-shared crop cycling

HYPOTHESIS: Deterministically cycling through every full and central crop offset will exceed 9,247 correct predictions by matching the uniform validation ensemble more precisely while retaining beneficial batch-correlated augmentation.

INTENDED_EDIT: Replace random batch-wide crop draws with balanced, coprime cycles covering all 25 full offsets and all 9 central offsets.

EVIDENCE: Per-example translation sampling regressed to 9,219, indicating batch-shared offsets are beneficial; this preserves that structure and the proven uniform crop distribution while eliminating finite-run sampling imbalance.

<<<<<<< SEARCH
    offsets = torch.randint(0, 5, (2,))
    offset_y, offset_x = int(offsets[0]), int(offsets[1])
    central_offsets = torch.randint(1, 4, (2,))
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
=======
    full_index = (7 * step) % 25
    offset_y, offset_x = divmod(full_index, 5)
    central_index = (5 * step) % 9
    central_y, central_x = divmod(central_index, 3)
    central_y += 1
    central_x += 1
>>>>>>> REPLACE