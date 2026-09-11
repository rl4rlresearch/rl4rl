MECHANISM: Cardinalized one-pixel translation augmentation

HYPOTHESIS: Reassigning diagonal one-pixel shifts to cardinal shifts will exceed 9,196 correct predictions by eliminating unsupported diagonal transforms while closely matching the successful validation ensemble’s center-to-cardinal exposure ratio.

INTENDED_EDIT: Detect diagonally shifted training samples and randomly center one shifted axis, changing augmentation from 25% center/50% cardinal/25% diagonal to 25% center/75% cardinal/0% diagonal.

EVIDENCE: Restricting translations from ±2 to ±1 improved validation correctness from 9,166 to 9,196, while adding diagonal validation views previously reduced it to 9,155; the prior diagonal-removal attempts timed out and therefore did not provide contrary accuracy evidence.

<<<<<<< SEARCH
    offset_x = (
        torch.randint(0, 2, (images.shape[0],), device=images.device)
        + torch.randint(0, 2, (images.shape[0],), device=images.device)
        + 1
    )
    patches = padded.unfold(2, 28, 1).unfold(3, 28, 1)
=======
    offset_x = (
        torch.randint(0, 2, (images.shape[0],), device=images.device)
        + torch.randint(0, 2, (images.shape[0],), device=images.device)
        + 1
    )
    diagonal = (offset_y != 2) & (offset_x != 2)
    keep_vertical = torch.rand(images.shape[0], device=images.device) < 0.5
    offset_y = torch.where(
        diagonal & ~keep_vertical,
        torch.full_like(offset_y, 2),
        offset_y,
    )
    offset_x = torch.where(
        diagonal & keep_vertical,
        torch.full_like(offset_x, 2),
        offset_x,
    )
    patches = padded.unfold(2, 28, 1).unfold(3, 28, 1)
>>>>>>> REPLACE