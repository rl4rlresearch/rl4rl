MECHANISM: Zero-sum-head-compensated recurrent compression

HYPOTHESIS: A 97-unit GRU with the seven-output zero-sum classifier will achieve at least 85% validation accuracy while reducing recurrent MACs by approximately 1.86% versus the current 98-unit model.

INTENDED_EDIT: Reduce the GRU hidden state and temporal summary from 98 to 97 units, retaining all 32 frames and classifying from the complete 97-dimensional summary.

EVIDENCE: The 97-unit model with a conventional eight-output head narrowly missed at 84.42%, while changing the 98-unit model from an eight-output head to the seven-output zero-sum parameterization improved accuracy from 85.52% to 86.26%; a comparable gain would place the compressed model above 85%.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 7)
=======
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.classifier(averaged[:, :97])
=======
        logits = self.classifier(averaged)
>>>>>>> REPLACE