MECHANISM: Smaller-batch optimization rescue at the recurrent-width boundary

HYPOTHESIS: A 115-unit GRU trained with batch size 64 will recover at least 85% validation accuracy through roughly twice as many optimizer updates while retaining the previously measured 760,672,100 inference MACs.

INTENDED_EDIT: Reduce the qualified recurrent width from 116 to 115 units and halve the training batch size without changing inference structure, augmentation, loss, or learning-rate schedule.

EVIDENCE: The 116-unit model qualified at 85.77%, while the adjacent 115-unit model reached 84.17% using batch size 128; retraining that exact lower-cost width with more optimizer updates directly tests whether the narrow miss was optimization-limited rather than capacity-limited.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 116, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
        self.endpoint_classifier = nn.Linear(116, 8)
=======
        self.gru = nn.GRU(20, 115, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(115, 8)
        self.endpoint_classifier = nn.Linear(115, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 116, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 116, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 115, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 115, device=device, dtype=dtype)
>>>>>>> REPLACE