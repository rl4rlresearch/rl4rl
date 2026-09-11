MECHANISM: One-unit recurrent-width frontier search

HYPOTHESIS: A 75-unit GRU with mean, final-state, and running-maximum readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2.3% versus the verified 76-unit model.

INTENDED_EDIT: Reduce the GRU and recurrent summary widths from 76 to 75 units and the classifier input from 228 to 225 features, preserving all 32 frames and the established training procedure.

EVIDENCE: The 76-unit model achieved 87.36% validation accuracy, 2.36 points above the requirement, after the 77-, 78-, and 79-unit models also qualified; this supports another minimal structural reduction at the current cost frontier.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 76, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(228, 8)
=======
        self.gru = nn.GRU(20, 75, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(225, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 76), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 75, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 75, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 75), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE