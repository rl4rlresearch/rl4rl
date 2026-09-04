MECHANISM: Penultimate-frame pruning with endpoint preservation

HYPOTHESIS: The 60-unit GRU processing frames 3–26 and frame 28 will retain at least 85% accuracy while reducing total inference MACs from 306,309,600 to approximately 294,573,600.

INTENDED_EDIT: Reduce execution from 26 to 25 recurrent steps by omitting frame 27 while preserving the informative frame-28 endpoint and the contiguous early sequence.

EVIDENCE: Frames 3–28 achieved 85.40%, while truncating to frames 3–27 fell to 84.17%; preserving frame 28 and pruning its predecessor is the most direct test of whether the endpoint matters more than one adjacent late frame. The failed frame-4 omission also argues against disrupting early continuity.

<<<<<<< SEARCH
        return list(range(3, available_frames - 3))
=======
        return [*range(3, available_frames - 5), available_frames - 4]
>>>>>>> REPLACE