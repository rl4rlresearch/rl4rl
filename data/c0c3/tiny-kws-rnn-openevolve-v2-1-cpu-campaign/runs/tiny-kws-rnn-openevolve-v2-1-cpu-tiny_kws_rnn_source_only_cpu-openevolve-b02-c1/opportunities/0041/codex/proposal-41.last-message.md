MECHANISM: Three-frame hierarchical gated memory

HYPOTHESIS: Updating the 64-unit slow GRU every third frame instead of every second frame will retain at least 85% validation accuracy while reducing total inference MACs from 454,743,920 to approximately 384,640,880.

INTENDED_EDIT: Preserve the successful 48-unit fast GRU, 64-unit slow GRU, 25-frame schedule, features, readout, and training procedure, but reduce slow-memory execution from 12 to 8 updates per example.

EVIDENCE: The hierarchical two-rate model reached 86.38% accuracy while showing that the slow recurrent capacity need not execute at every frame; its 1.38-point margin supports testing a lower slow-state update frequency that removes four costly 64-unit GRU executions per example.

<<<<<<< SEARCH
    """Hierarchical gated memory with frame-rate and pair-rate state updates."""
=======
    """Hierarchical gated memory with frame-rate and three-frame state updates."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        pair_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 2
        ) == 1
        if bool(pair_boundary.any()):
            updated = self.slow_cell(
                fast[pair_boundary], slow[pair_boundary]
            )
            next_slow = slow.clone()
            next_slow[pair_boundary] = updated
=======
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 3
        ) == 2
        if bool(slow_boundary.any()):
            updated = self.slow_cell(
                fast[slow_boundary], slow[slow_boundary]
            )
            next_slow = slow.clone()
            next_slow[slow_boundary] = updated
>>>>>>> REPLACE