MECHANISM: One-channel recurrent-width trim under smaller-batch training

HYPOTHESIS: A 116-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact inference MACs by approximately 1.6% versus the passing 117-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 117 to 116 while preserving the successful 23-frame schedule and batch-64 training procedure.

EVIDENCE: Batch-64 training raised the 117-unit, 23-step model from 84.908% to 86.503%; this margin motivates retesting the adjacent lower width under the improved optimization regime.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 117, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(117, 8)
=======
        self.gru = nn.GRU(20, 116, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 117, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 117, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 116, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 116, device=device, dtype=dtype)
>>>>>>> REPLACE