MECHANISM: Temporal-max-assisted recurrent width reduction

HYPOTHESIS: An 84-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing exact recurrent MACs by about 3.8% versus the verified 86-unit model.

INTENDED_EDIT: Reduce the GRU and all recurrent summary widths from 86 to 84 units, and reduce the classifier input from 258 to 252 features.

EVIDENCE: Adding temporal maximum pooling raised the 86-unit model from 84.66% to 86.87%, creating a 1.87-point margin; this supports a conservative two-unit structural reduction while preserving all 32 frames and the successful readout.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(258, 8)
=======
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 86), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 84), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE