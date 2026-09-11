MECHANISM: Front-loaded smooth activation with faster deep ReLU blocks

HYPOTHESIS: Keeping the first GELU while replacing the two deeper GELUs with ReLU will retain at least 9,258 correct predictions and reduce training time below 75 seconds.

INTENDED_EDIT: Preserve the accuracy-critical first GELU and use ReLU only after the second and third BatchNorm layers.

EVIDENCE: Replacing all GELUs with ReLU reduced runtime to 68.65 seconds, while replacing only the first GELU produced nearly the same accuracy loss as replacing all three (9,233 versus 9,235 correct) and retained the slow 80.27-second runtime. This isolates the first GELU as accuracy-sensitive and the deeper GELUs as the likely runtime cost.

<<<<<<< SEARCH
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
>>>>>>> REPLACE