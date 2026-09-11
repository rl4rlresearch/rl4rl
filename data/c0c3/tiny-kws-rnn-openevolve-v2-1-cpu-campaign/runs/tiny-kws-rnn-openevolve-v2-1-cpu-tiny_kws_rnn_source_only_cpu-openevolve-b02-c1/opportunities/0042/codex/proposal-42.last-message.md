MECHANISM: Four-frame hierarchical gated memory

HYPOTHESIS: Updating the 64-unit slow GRU every fourth frame will retain at least 85% validation accuracy while reducing total inference MACs from 384,640,880 to approximately 349,589,360.

INTENDED_EDIT: Preserve the successful features, 48-unit fast GRU, 64-unit slow GRU, 25-frame schedule, readout, and training procedure, but reduce slow-memory execution from eight to six updates per example.

EVIDENCE: Moving from 12 to eight slow updates retained 85.77% accuracy and removed roughly 70.1 million MACs; the remaining 0.77-point margin motivates the smaller next reduction of two slow updates.

<<<<<<< SEARCH
    """Hierarchical gated memory with frame-rate and three-frame state updates."""
=======
    """Hierarchical gated memory with frame-rate and four-frame state updates."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 3
        ) == 2
=======
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 4
        ) == 3
>>>>>>> REPLACE