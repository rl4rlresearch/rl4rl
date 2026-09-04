MECHANISM: Stratified terminal translation sampling

HYPOTHESIS: On the verified 39/64 curriculum, balancing the five inference-aligned translation choices within every terminal minibatch will exceed 9,167 correct predictions by reducing augmentation-distribution noise without reducing view diversity.

INTENDED_EDIT: Restore the best verified 39/64 transition and replace independent terminal direction draws with a rotating, near-uniform assignment across each batch.

EVIDENCE: Reference Design 1 achieved the best completed result—9,167 correct—with the 39/64 boundary, while later transitions performed worse. The terminal phase and inference both use the same five translation directions, motivating exact minibatch stratification as a focused improvement.

<<<<<<< SEARCH
    if step * 4 < total_steps * 3:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE

<<<<<<< SEARCH
        directions = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
=======
        directions = (batch_indices + step) % 5
>>>>>>> REPLACE