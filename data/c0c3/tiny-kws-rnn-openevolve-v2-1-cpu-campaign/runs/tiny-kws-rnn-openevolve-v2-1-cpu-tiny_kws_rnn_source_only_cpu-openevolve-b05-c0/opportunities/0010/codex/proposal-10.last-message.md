MECHANISM: Recurrent-width scaling with dual temporal readout

HYPOTHESIS: An 80-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 15.8% versus the verified 88-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 88 to 80 and resize the classifier and recurrent summary state accordingly.

EVIDENCE: The 88-unit model achieved 88.22% accuracy after successful reductions from 104 and 96 units, leaving a 3.22-point margin and showing that width reduction with full temporal coverage is more reliable than frame subsampling.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(160, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
>>>>>>> REPLACE