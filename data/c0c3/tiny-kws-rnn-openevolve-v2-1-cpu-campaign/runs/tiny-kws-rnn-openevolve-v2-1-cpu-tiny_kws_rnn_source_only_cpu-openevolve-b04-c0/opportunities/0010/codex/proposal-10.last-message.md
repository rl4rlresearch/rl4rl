MECHANISM: Fine-grained recurrent-width reduction

HYPOTHESIS: An 86-unit GRU processing all 32 frames with the dual mean/final-state readout will retain at least 85% validation accuracy while reducing recurrent MACs by about 4.1% versus the verified 88-unit model.

INTENDED_EDIT: Reduce the GRU hidden state and temporal summary from 88 to 86 units, and reduce the classifier input from 176 to 172 features.

EVIDENCE: The 88-unit full-sequence model achieved 85.77% accuracy, leaving a 0.77-point margin; a two-unit reduction is the smallest practical structural step and preserves the temporal coverage whose reduction previously hurt accuracy.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(172, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE