MECHANISM: Twelve-frame hierarchical gated memory

HYPOTHESIS: Updating the 64-unit slow GRU at processed steps 12 and 24 will retain at least 85% validation accuracy while reducing slow updates from three to two and total inference MACs from 297,012,080 to approximately 279,486,320.

INTENDED_EDIT: Preserve the verified features, 48-unit fast GRU, 64-unit slow GRU, 25-frame schedule, readout, and training procedure while increasing the slow-memory update interval from eight frames to twelve.

EVIDENCE: Progressively reducing slow updates from twelve to three retained the accuracy target at every tested interval; the current three-update model achieves 85.64%, and removing one slow update has consistently saved exactly 17,525,760 total MACs.

<<<<<<< SEARCH
    """Hierarchical gated memory with frame-rate and eight-frame state updates."""
=======
    """Hierarchical gated memory with frame-rate and twelve-frame state updates."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 8
        ) == 7
=======
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 12
        ) == 11
>>>>>>> REPLACE