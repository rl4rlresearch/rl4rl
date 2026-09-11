MECHANISM: Coverage-preserving 24-step temporal resampling

HYPOTHESIS: Redistributing 24 recurrent steps across 27 uniformly sampled candidates while retaining frame 4 will achieve at least 85% validation accuracy at approximately 336.06M total inference MACs.

INTENDED_EDIT: Use a 27-frame base schedule and remove the first three densely spaced nonzero frames, producing 24 steps with broader temporal coverage instead of creating the failed frame-0-to-frame-5 gap.

EVIDENCE: The 25-step schedule passed at 86.63%, whereas deleting a fourth adjacent early frame fell to 84.17%; this isolates the enlarged initial sampling gap as a plausible failure mode and motivates testing 24 steps through redistribution.

<<<<<<< SEARCH
        base_steps = min(28, available_frames)
        indices = [
            i * (available_frames - 1) // (base_steps - 1)
            for i in range(base_steps)
        ]
        if len(indices) > 27:
            del indices[1:4]
=======
        base_steps = min(27, available_frames)
        indices = [
            i * (available_frames - 1) // (base_steps - 1)
            for i in range(base_steps)
        ]
        if len(indices) > 26:
            del indices[1:4]
>>>>>>> REPLACE