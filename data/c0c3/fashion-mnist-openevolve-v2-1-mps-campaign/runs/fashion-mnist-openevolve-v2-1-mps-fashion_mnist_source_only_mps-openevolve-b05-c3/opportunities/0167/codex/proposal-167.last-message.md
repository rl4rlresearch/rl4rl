MECHANISM: Center-biased terminal fine-tuning with argmax-invariant ensemble sharpening

HYPOTHESIS: Restoring the verified 1/3 terminal centered-crop rate will recover 9,210 correct predictions, while 1.15× final sharpening preserves those argmax decisions and lowers cross-entropy below 0.2253669.

INTENDED_EDIT: Sample two centered and four cardinal-shift outcomes during terminal training, then increase the eight-view ensemble’s final log-probability multiplier from 1.10× to 1.15×.

EVIDENCE: The exact 1/3 terminal sampler achieved the best qualified correctness of 9,210, whereas 3/7 fell to 9,208; increasing sharpening from 1.05× to 1.10× previously preserved correctness and reduced cross-entropy, while the runtime-neutral 1.15× test remains unresolved.

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.10)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.15)
>>>>>>> REPLACE

<<<<<<< SEARCH
        directions = torch.randint(
            0, 7, (images.shape[0],), device=images.device
        )
=======
        directions = torch.randint(
            0, 6, (images.shape[0],), device=images.device
        )
>>>>>>> REPLACE