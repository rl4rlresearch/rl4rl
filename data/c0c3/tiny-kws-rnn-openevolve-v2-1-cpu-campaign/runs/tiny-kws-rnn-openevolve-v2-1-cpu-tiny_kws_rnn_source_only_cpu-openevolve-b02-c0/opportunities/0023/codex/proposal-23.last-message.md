MECHANISM: Symmetric rank-seven classifier factorization

HYPOTHESIS: Factoring the 99-to-8 classifier through seven learned logits will retain at least 85% validation accuracy while saving 43 MACs and 43 parameters per classifier, because eight-class softmax needs only seven independent logit-difference dimensions.

INTENDED_EDIT: Replace the direct classifier with a bias-free 99-to-7 projection followed by a learned 7-to-8 decoder, preserving access to all 99 recurrent channels and trainable logits for every class.

EVIDENCE: The fixed-reference seven-logit classifier reached only 83.56%, showing that asymmetric treatment of one class harms learning; the proposed symmetric decoder retains the same structural dimensionality reduction while avoiding a fixed class, and unlike the failed 98-channel readout it discards no recurrent-state information.

<<<<<<< SEARCH
        self.classifier = nn.Linear(99, 8)
=======
        self.classifier = nn.Sequential(
            nn.Linear(99, 7, bias=False),
            nn.Linear(7, 8),
        )
>>>>>>> REPLACE