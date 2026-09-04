MECHANISM: Seven-bin trailing-boundary pruning

HYPOTHESIS: Processing frames 1–28 in seven four-step bins will retain at least 85% validation accuracy while reducing execution to 28 recurrent steps and total inference cost to approximately 614.5 million MACs.

INTENDED_EDIT: Prune frames 0 and 29–31, replace the eight-bin state/readout with seven complete four-step bins, and resize the classifier accordingly.

EVIDENCE: The 29-step 45/45/46 model achieved 87.12% accuracy at 637,155,590 MACs; its 2.12-point margin supports removing the adjacent trailing frame, whose removal also makes seven complete bins more structurally efficient than retaining an empty eighth bin.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three balanced GRUs with an eight-bin causal temporal readout."""
=======
class KeywordGRU(nn.Module):
    """Three balanced GRUs with a seven-bin causal temporal readout."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Linear(8 * 136, 8)
=======
        self.classifier = nn.Linear(7 * 136, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bins = torch.zeros(batch_size, 8, 136, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 8, device=device, dtype=dtype)
=======
        bins = torch.zeros(batch_size, 7, 136, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 7, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=7)
        assignment = F.one_hot(bin_index[:, 0], num_classes=8).to(output.dtype)
=======
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=6)
        assignment = F.one_hot(bin_index[:, 0], num_classes=7).to(output.dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bin_indices = (positions // 4).clamp(max=7)
        assignments = F.one_hot(bin_indices, num_classes=8).to(outputs.dtype)
=======
        bin_indices = (positions // 4).clamp(max=6)
        assignments = F.one_hot(bin_indices, num_classes=7).to(outputs.dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 5:
            return list(range(available_frames))
        return list(range(1, available_frames - 3))
>>>>>>> REPLACE