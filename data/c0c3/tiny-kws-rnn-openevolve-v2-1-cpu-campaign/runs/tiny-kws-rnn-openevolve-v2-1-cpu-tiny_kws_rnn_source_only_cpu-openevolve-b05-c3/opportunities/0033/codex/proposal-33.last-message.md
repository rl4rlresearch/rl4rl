MECHANISM: Smaller-batch adjacent recurrent-width probe

HYPOTHESIS: A 111-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 724,398,080 to approximately 712,502,340.

INTENDED_EDIT: Reduce the recurrent state and both readout widths from 120 to 111 units and use the proven batch-64 training procedure, preserving the qualified 20-frame schedule and all other training choices.

EVIDENCE: The adjacent 112-unit design achieved 87.61% accuracy at 724,398,080 MACs, leaving a 2.61-point margin; testing 111 units is the most informative remaining width-boundary probe.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)
        self.endpoint_classifier = nn.Linear(120, 8)
=======
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(111, 8)
        self.endpoint_classifier = nn.Linear(111, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
>>>>>>> REPLACE