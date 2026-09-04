MECHANISM: Throughput-assisted recurrent-width reduction

HYPOTHESIS: A 58-unit GRU trained with batch size 96 will retain at least 85% validation accuracy, finish within the verification time limit, and reduce total inference MACs by approximately 5.7% versus the verified 60-unit model.

INTENDED_EDIT: Reduce the GRU and temporal-summary width from 60 to 58, resize the classifier accordingly, and increase training batch size from 64 to 96 to reduce optimizer-step overhead.

EVIDENCE: The 58-unit, batch-64 run reached 88.47% accuracy and 354,714,080 inference MACs, but its 207.6-second training time exceeded the limit; prior larger-batch runs completed substantially faster.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 96
GRAD_CLIP_NORM = 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
=======
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 60, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
>>>>>>> REPLACE