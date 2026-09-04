MECHANISM: Late dense-cluster temporal pruning

HYPOTHESIS: Omitting frame 26 from the qualified 21-step schedule will retain at least 85% validation accuracy while reducing execution to 20 recurrent steps and total inference MACs to approximately 264,555,520.

INTENDED_EDIT: Add frame 26 to the skipped set for 32-frame inputs, preserving frame 29 and both endpoints.

EVIDENCE: The 21-step schedule qualified at 85.40% by pruning the center of the dense 5–7 cluster; frames 3 and 10 failed as additional omissions, while frame 29 was already shown accuracy-critical, motivating the analogous late dense 25–27 cluster.

<<<<<<< SEARCH
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
=======
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(13 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
                skipped.add(3 * available_frames // 16)
>>>>>>> REPLACE