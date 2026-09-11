MECHANISM: Single late slow-memory update

HYPOTHESIS: Updating the 64-unit slow GRU only at processed step 24 will retain at least 85% validation accuracy while reducing slow updates from two to one and total inference MACs from 279,486,320 to approximately 261,960,560.

INTENDED_EDIT: Preserve the verified features, 48-unit fast GRU, 64-unit slow GRU, 25-frame schedule, readout, and training procedure while increasing the slow-memory update interval from twelve to twenty-four frames.

EVIDENCE: Every tested reduction from twelve slow updates down to two retained the accuracy target; the current two-update model achieves 85.40%, and each removed slow update has consistently saved exactly 17,525,760 total MACs.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Hierarchical gated memory with frame-rate and twelve-frame state updates."""
=======
class KeywordGRU(nn.Module):
    """Hierarchical gated memory with frame-rate and one late slow update."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 12
        ) == 11
=======
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 24
        ) == 23
>>>>>>> REPLACE