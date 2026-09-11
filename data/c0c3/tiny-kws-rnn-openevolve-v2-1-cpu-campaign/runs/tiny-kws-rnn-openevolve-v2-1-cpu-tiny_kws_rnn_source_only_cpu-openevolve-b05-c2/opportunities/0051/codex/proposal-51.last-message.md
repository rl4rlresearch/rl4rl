MECHANISM: Late-cluster edge pruning

HYPOTHESIS: Omitting frame 27 instead of frame 26 from the qualified 21-step schedule will preserve at least 85% accuracy while reducing execution to 20 recurrent steps and total inference MACs to approximately 264,555,520.

INTENDED_EDIT: Restore the qualified frame-6 omission and additionally omit frame 27 for 32-frame inputs, retaining frame 26, accuracy-critical frame 29, and both endpoints.

EVIDENCE: The frame-26 20-step design reached 84.91%, the closest failed temporal reduction; pruning adjacent cluster-edge frame 27 is the smallest informative schedule substitution while preserving the same compute reduction.

<<<<<<< SEARCH
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
=======
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
                skipped.add(27 * available_frames // 32)
>>>>>>> REPLACE