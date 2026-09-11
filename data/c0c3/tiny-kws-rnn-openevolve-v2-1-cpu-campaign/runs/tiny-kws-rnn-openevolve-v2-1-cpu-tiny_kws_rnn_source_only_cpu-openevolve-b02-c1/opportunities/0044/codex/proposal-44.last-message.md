MECHANISM: Six-frame hierarchical gated memory

HYPOTHESIS: Updating the 64-unit slow GRU every sixth frame will retain at least 85% validation accuracy while reducing slow updates from five to four and total inference MACs from 332,063,600 to approximately 314,537,840.

INTENDED_EDIT: Preserve the verified features, 48-unit fast GRU, 64-unit slow GRU, 25-frame schedule, readout, and training procedure while increasing the slow-memory update interval from five frames to six.

EVIDENCE: Increasing the interval from four to five frames reduced total inference MACs by 17,525,760 while improving validation accuracy from 86.26% to 86.87%; the resulting 1.87-point margin motivates removing one more slow update through the next controlled interval increase.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Hierarchical gated memory with frame-rate and five-frame state updates."""
=======
class KeywordGRU(nn.Module):
    """Hierarchical gated memory with frame-rate and six-frame state updates."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 5
        ) == 4
=======
        slow_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 6
        ) == 5
>>>>>>> REPLACE