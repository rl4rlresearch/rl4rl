MECHANISM: Max-supported recurrent width compression

HYPOTHESIS: An 83-unit GRU with the verified 26-frame mean-max-terminal readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 2.2% versus the successful 84-unit model.

INTENDED_EDIT: Reduce the GRU and all recurrent summary widths from 84 to 83, and resize the classifier input from 252 to 249 while preserving the frame schedule and training procedure.

EVIDENCE: The 84-unit/26-step model achieved 86.63% accuracy, providing substantial margin, and the prior 83-unit/29-step model achieved 86.01%; together these results support a one-unit width reduction while retaining the accuracy-restoring max readout.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
=======
        self.gru = nn.GRU(20, 83, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(249, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 84, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 83, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 83, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 83, device=device, dtype=dtype)
>>>>>>> REPLACE