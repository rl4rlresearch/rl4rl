MECHANISM: Continued recurrent-width reduction with salience-preserving temporal readout

HYPOTHESIS: A 76-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 8.8% versus the verified 80-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 80 to 76, resize its recurrent state tensors, and reduce the three-way classifier input from 240 to 228 features while preserving all 32 causal frames and the training procedure.

EVIDENCE: Adding maximum-output pooling allowed the 84-unit model to reach 86.75%, and the subsequent 80-unit reduction still achieved 86.50%; the retained 1.50-point margin supports continuing the established four-unit structural width search.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 80, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
=======
        self.gru = nn.GRU(20, 76, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(228, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 80, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 80, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 76, device=device, dtype=dtype)
>>>>>>> REPLACE