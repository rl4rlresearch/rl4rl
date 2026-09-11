MECHANISM: Moderate hidden-width reduction at full temporal resolution

HYPOTHESIS: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 13% versus the verified 112-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 112 to 104 while preserving the successful full-frame training procedure.

EVIDENCE: The 112-unit full-frame model achieved 86.13% accuracy, while temporal subsampling failed; this motivates preserving all 32 frames and testing the next meaningful reduction along the successful width-efficiency axis.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
>>>>>>> REPLACE