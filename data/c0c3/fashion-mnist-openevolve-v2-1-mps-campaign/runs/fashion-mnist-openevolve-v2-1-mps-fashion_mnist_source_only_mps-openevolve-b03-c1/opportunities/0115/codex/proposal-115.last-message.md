MECHANISM: Triangular one-pixel translation augmentation

HYPOTHESIS: Restricting training translations from ±2 to ±1 pixels will exceed 9,166 correct predictions by reducing transform mismatch and preserving more discriminative edge detail during the fixed two-pass exposure.

INTENDED_EDIT: Change both training crop offsets to a triangular −1/0/+1 distribution while retaining diagonal shifts, flips, and all model and optimization settings.

EVIDENCE: The successful validation ensemble uses only ±1 cardinal shifts, while adding diagonal views reduced correctness from 9,166 to 9,155; isolating shift magnitude tests whether the broader ±2 training support is excessive without confounding architecture or runtime.

<<<<<<< SEARCH
    offset_y = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
    offset_x = torch.randint(
        0, 3, (images.shape[0],), device=images.device
    ) + torch.randint(0, 3, (images.shape[0],), device=images.device)
=======
    offset_y = (
        torch.randint(0, 2, (images.shape[0],), device=images.device)
        + torch.randint(0, 2, (images.shape[0],), device=images.device)
        + 1
    )
    offset_x = (
        torch.randint(0, 2, (images.shape[0],), device=images.device)
        + torch.randint(0, 2, (images.shape[0],), device=images.device)
        + 1
    )
>>>>>>> REPLACE