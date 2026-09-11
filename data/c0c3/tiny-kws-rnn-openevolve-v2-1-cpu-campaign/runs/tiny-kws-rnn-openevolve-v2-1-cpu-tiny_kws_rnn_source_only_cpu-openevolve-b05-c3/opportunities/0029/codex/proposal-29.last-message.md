MECHANISM: Smaller-batch recurrent-width boundary probe

HYPOTHESIS: A 114-unit GRU trained with batch size 64 will retain at least 85% validation accuracy while reducing total inference MACs from 760,672,100 to approximately 748,482,960.

INTENDED_EDIT: Reduce the recurrent and readout widths from 118 to 114 units and halve the batch size to provide roughly twice as many optimizer updates.

EVIDENCE: The 115-unit model improved from 84.17% with batch size 128 to 86.87% with batch size 64 at identical inference cost, leaving a 1.87-point margin and motivating the adjacent 114-unit boundary test under the proven smaller-batch procedure.

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
        self.endpoint_classifier = nn.Linear(118, 8)
=======
        self.gru = nn.GRU(20, 114, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)
        self.endpoint_classifier = nn.Linear(114, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 118, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 118, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 114, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 114, device=device, dtype=dtype)
>>>>>>> REPLACE