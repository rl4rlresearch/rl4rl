MECHANISM: Recurrent width reduction with dual-timescale readout

HYPOTHESIS: An 84-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 8.1% versus the verified 88-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 88 to 84 and resize the classifier and recurrent state tensors accordingly, while preserving the full 32-frame schedule and training procedure.

EVIDENCE: The 88-unit dual-readout model achieved 85.77% accuracy with 744,740,480 MACs, and every tested dual-readout width from 92 through 100 exceeded the accuracy requirement; the next four-unit reduction directly continues the established width search.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
>>>>>>> REPLACE