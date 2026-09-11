MECHANISM: Smaller-batch optimization rescue

HYPOTHESIS: Doubling the optimizer updates with batch size 64 will let the narrowly failing 117-unit, 23-step GRU reach at least 85% accuracy while reducing inference MACs by approximately 1.6% versus the passing 118-unit model.

INTENDED_EDIT: Reduce recurrent/readout width from 118 to 117 and training batch size from 128 to 64, preserving the verified schedule, loss, augmentation, optimizer, and learning-rate schedule.

EVIDENCE: The identical 117-unit, 23-step architecture missed the threshold by only one validation example at 84.908%, while it passed at 24 and 25 steps; smaller batches provide twice as many parameter updates within the fixed 50,000-example exposure without adding inference cost.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 118, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(118, 8)
=======
        self.gru = nn.GRU(20, 117, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(117, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 118, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 118, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 117, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 117, device=device, dtype=dtype)
>>>>>>> REPLACE