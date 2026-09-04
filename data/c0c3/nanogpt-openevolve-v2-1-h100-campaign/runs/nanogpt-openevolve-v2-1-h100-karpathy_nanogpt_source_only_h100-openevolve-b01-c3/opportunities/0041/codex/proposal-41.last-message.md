MECHANISM: Direction-aware MLP Muon compensation

HYPOTHESIS: Compensating only the expanding MLP matrices will retain at least 445M training tokens and reduce val_bpb below 0.982780 by preserving the beneficial aspect correction without unnecessarily lowering the contraction matrices’ learning rate.

INTENDED_EDIT: Apply the sqrt(4.375/5) Muon-rate adjustment only to 2560×512 MLP expansion weights; leave 512×2560 projection weights at the proven base rate.

EVIDENCE: Aspect compensation improved the 5× MLP from 0.982905 to 0.982780, but the fused optimizer applies its aspect-ratio multiplier only when rows exceed columns. The current symmetric shape test therefore also reduces the projection rate even though that matrix receives no corresponding optimizer-side increase.

<<<<<<< SEARCH
            # The fused Muon step scales LR by sqrt(matrix aspect ratio).
            # Compensate 5x MLP matrices back to the effective rate tuned at 4.375x.
            mlp_lr_scale = math.sqrt(7 / 8) if max(shape) == 5 * min(shape) else 1.0
=======
            # The fused Muon step scales LR only when rows exceed columns.
            # Compensate the 5x expansion matrix; the contraction matrix is unscaled.
            mlp_lr_scale = math.sqrt(7 / 8) if shape[-2] == 5 * shape[-1] else 1.0
>>>>>>> REPLACE