MECHANISM: Sparse tail sampling with preserved temporal coverage

HYPOTHESIS: Processing frames 2–23, 25, 27, and 29 will achieve at least 85% validation accuracy while reducing execution from 26 to 25 recurrent steps.

INTENDED_EDIT: Replace the contiguous 26-frame schedule with 25 frames spanning the stronger frames-2–29 window, omitting alternating late frames.

EVIDENCE: Frames 2–29 achieved 86.50%, while contiguous frames 2–26 fell to 83.31%; preserving the later endpoint and broader temporal span directly tests whether coverage, rather than the 25-step count itself, caused that failure.

<<<<<<< SEARCH
        if available_frames >= 6:
            return list(range(2, available_frames - 4))
        return list(range(available_frames))
=======
        if available_frames >= 8:
            full_window = list(range(2, available_frames - 2))
            return full_window[:-6] + full_window[-5::2]
        return list(range(available_frames))
>>>>>>> REPLACE