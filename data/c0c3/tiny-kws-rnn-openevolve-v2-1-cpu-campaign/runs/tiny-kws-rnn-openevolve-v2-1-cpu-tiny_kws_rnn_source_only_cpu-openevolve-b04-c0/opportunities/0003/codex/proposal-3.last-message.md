MECHANISM: Conservative recurrent-width reduction

HYPOTHESIS: A 104-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs by roughly 12% relative to the verified 112-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier input width from 112 to 104 while preserving all 32 causal frames and the proven training procedure.

EVIDENCE: Reducing width from 128 to 112 preserved 86.26% accuracy and cut recurrent MACs to 1,156,700,160, while temporal decimation failed at 81.72%; the remaining 1.26-point margin supports a smaller, conservative width reduction without discarding frames.

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