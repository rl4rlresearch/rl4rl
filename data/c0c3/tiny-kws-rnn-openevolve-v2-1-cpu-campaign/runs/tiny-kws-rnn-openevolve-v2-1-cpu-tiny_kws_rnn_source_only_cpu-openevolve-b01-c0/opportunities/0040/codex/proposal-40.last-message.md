MECHANISM: One-channel recurrent-width trim under batch-64 training

HYPOTHESIS: A 115-unit, 23-step GRU trained with batch size 64 will achieve at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 116-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 116 to 115 while preserving the successful schedule and training procedure.

EVIDENCE: The 116-unit model achieved 86.380% accuracy at batch size 64, leaving a 1.38-point margin; the preceding 117-to-116 trim also retained accuracy while lowering MACs, motivating the next isolated width-boundary test.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 116, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
=======
        self.gru = nn.GRU(20, 115, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(115, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 116, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 116, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 115, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 115, device=device, dtype=dtype)
>>>>>>> REPLACE