MECHANISM: Onset-preserving boundary frame pruning

HYPOTHESIS: An 86-unit GRU retaining the early frame lost by the failed 26-step design and instead pruning the penultimate frame will achieve at least 85% accuracy at 580.63M total inference MACs.

INTENDED_EDIT: Increase the GRU/readout width from 85 to 86 and create a 26-frame schedule by removing the already-proven redundant second frame plus the penultimate frame from the successful 28-frame base schedule.

EVIDENCE: The 85-unit/27-step schedule passed at 86.63%, while the 86-unit/26-step design that removed both earliest nonzero frames missed the threshold by only 0.09 points; preserving potential onset information while pruning a likely trailing-silence frame directly tests whether pruning location caused that failure.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(172, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if len(indices) > 27:
            del indices[1]
=======
        if len(indices) > 26:
            del indices[-2]
            del indices[1]
>>>>>>> REPLACE