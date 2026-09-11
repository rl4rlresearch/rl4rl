MECHANISM: Five-frame hierarchical gated memory

HYPOTHESIS: Updating the 64-unit slow GRU every fifth frame will retain at least 85% validation accuracy while reducing slow updates from six to five and total inference MACs from 349,589,360 to approximately 332,063,600.

INTENDED_EDIT: Preserve the verified features, 48-unit fast GRU, 64-unit slow GRU, 25-frame schedule, readout, and training procedure while changing the slow-memory update interval from four frames to five.

EVIDENCE: Moving from eight to six slow updates improved accuracy from 85.77% to 86.26% while removing 35,051,520 MACs; the resulting 1.26-point margin supports testing one additional slow-update reduction.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Hierarchical gated memory with frame-rate and four-frame state updates."""
=======
class KeywordGRU(nn.Module):
    """Hierarchical gated memory with frame-rate and five-frame state updates."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 4
        ) == 3
=======
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 5
        ) == 4
>>>>>>> REPLACE