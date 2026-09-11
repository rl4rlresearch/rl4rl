MECHANISM: Nested frame pruning with width-for-time trade

HYPOTHESIS: An 86-unit GRU processing 26 frames formed by removing one additional early frame from the successful 85-unit/27-step schedule will retain at least 85% accuracy while reducing total inference MACs by approximately 1.6%.

INTENDED_EDIT: Increase GRU and readout width from 85 to 86, resize the classifier accordingly, and prune the first two nonzero frames from the verified 28-frame base schedule.

EVIDENCE: The nested 85-unit/27-step design achieved 86.63% accuracy, substantially outperforming uniform 27-frame resampling; this supports preserving its frame alignment while making the next incremental width-for-time trade.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(172, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if len(indices) > 27:
            del indices[1]
=======
        if len(indices) > 26:
            del indices[1:3]
>>>>>>> REPLACE