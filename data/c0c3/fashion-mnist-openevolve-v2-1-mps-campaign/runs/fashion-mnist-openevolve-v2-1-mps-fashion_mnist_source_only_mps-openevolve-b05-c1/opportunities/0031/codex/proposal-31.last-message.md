MECHANISM: Zero-gamma residual initialization

HYPOTHESIS: Initializing each residual branch as an exact zero contribution will exceed 9,249 correct predictions by improving early optimization stability within the fixed two-pass training budget.

INTENDED_EDIT: Zero-initialize the final BatchNorm scale in every residual block while preserving architecture, parameter count, optimizer, augmentation, and inference.

EVIDENCE: Smaller batches improved correctness from 9,204 to 9,239 and the higher peak learning rate gained another prediction, indicating optimization-limited training; identity-starting residual blocks directly target optimization without adding runtime or regularization.

<<<<<<< SEARCH
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
=======
        self.bn2 = nn.BatchNorm2d(out_channels)
        nn.init.zeros_(self.bn2.weight)
        if in_channels == out_channels:
>>>>>>> REPLACE