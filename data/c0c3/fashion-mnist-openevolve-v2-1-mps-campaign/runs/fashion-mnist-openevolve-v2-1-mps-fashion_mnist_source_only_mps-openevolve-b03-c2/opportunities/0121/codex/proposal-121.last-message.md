MECHANISM: Interleaved shift-augmentation curriculum

HYPOTHESIS: Distributing the same 782 augmented batches across all 1,564 steps will exceed 9,256 correct predictions by preserving translation robustness during low-learning-rate refinement without changing total view exposure.

INTENDED_EDIT: Apply the existing five-view augmentation on alternating steps throughout training, retaining exactly the original number and ordering of augmented views.

EVIDENCE: Both increasing augmentation duration to 20/29 and decreasing it to 40% reduced validation_correct; this preserves the verified 50% exposure while testing whether its front-loaded timing is the limitation.

<<<<<<< SEARCH
    if step < total_steps // 2:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        view = step % 5
=======
    if step % 2 == 0:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        view = (step // 2) % 5
>>>>>>> REPLACE