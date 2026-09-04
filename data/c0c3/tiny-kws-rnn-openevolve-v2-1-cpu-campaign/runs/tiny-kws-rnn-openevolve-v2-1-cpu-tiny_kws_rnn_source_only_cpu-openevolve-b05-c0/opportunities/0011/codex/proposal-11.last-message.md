MECHANISM: Recurrent-width scaling with dual temporal readout

HYPOTHESIS: A 72-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 17.2% versus the verified 80-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 80 to 72 and resize the classifier and recurrent state tensors accordingly.

EVIDENCE: The 80-unit model achieved 88.22% accuracy after successive eight-unit reductions from 104 to 96, 88, and 80 units all remained above 87.9%, supporting another eight-unit reduction while preserving full temporal coverage.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(160, 8)
=======
        self.gru = nn.GRU(20, 72, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(144, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 72, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 72, device=device, dtype=dtype)
>>>>>>> REPLACE