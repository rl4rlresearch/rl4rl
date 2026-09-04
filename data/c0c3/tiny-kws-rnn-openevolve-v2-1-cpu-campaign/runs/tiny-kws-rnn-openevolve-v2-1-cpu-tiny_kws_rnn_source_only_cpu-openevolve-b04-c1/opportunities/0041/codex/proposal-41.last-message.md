MECHANISM: One-unit paired-GRU width pruning

HYPOTHESIS: A 69-unit paired-frame GRU will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 2.32%, from 246.11M to 240.40M.

INTENDED_EDIT: Reduce the paired GRU hidden width from 70 to 69 and resize its recurrent state and three-way pooled classifier accordingly.

EVIDENCE: The 70-unit paired-frame model achieved 85.77% accuracy with 13 learned transitions, providing a 0.77-point margin and motivating a conservative width-boundary probe that preserves all 26 input frames and the successful temporal mechanism.

<<<<<<< SEARCH
        self.gru = nn.GRU(40, 70, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(210, 8)
=======
        self.gru = nn.GRU(40, 69, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(207, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 70, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 70, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 70, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 69, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 69, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 69, device=device, dtype=dtype)
>>>>>>> REPLACE