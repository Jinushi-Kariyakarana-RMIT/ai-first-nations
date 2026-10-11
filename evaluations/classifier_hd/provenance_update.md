### Historical provenance recovered — 11 October 2026

The previous team report, [AI_FirstNations_178DSUG.pdf](../specifications/AI_FirstNations_178DSUG.pdf), provides historical dataset and performance claims that were absent from the current evaluation notebooks:

- Page 4 reports **477 labelled drone images**: Orange 156, Red 83, Yellow 238, from Darwin and Innisfail. Page 8 instead describes 425 images used for model development; the subset membership is not supplied.
- The category training script was committed by **Jake Dyl** in `8b418a5`. It points to `C:\Users\FSOS\Music\split_mangrove_images` and `C:\Users\FSOS\Music\internship\test`. Binary notebook paths point to `C:\Users\FSOS\Music\zow\man` and `...\non`.
- Both deployed checkpoints match historical Git bytes exactly: category checkpoint `694419e` (3 June 2026), binary checkpoint `5eef0d4` (5 June 2026). See `checkpoint_history.json` for hashes.
- The report claims **97% category accuracy**, but Table 8 supports total 4,080, Table 9 lists 32, and Figure 12 contains 22 images (7 Orange, 7 Red, 8 Yellow). These may refer to different experiments but are not reconciled. Table 8 also lists Orange precision 0.98, recall 0.97 and F1 0.87, which are arithmetically inconsistent. Treat all these as historical claims, not newly verified scores.
- Original code lists Orange/Red/Yellow and the report uses that order, strengthening the historical evidence for the current mapping. Training subsequently reads ImageFolder classes, however, and no original folder manifest or checkpoint class metadata was recovered; final semantic verification is still pending.
- **904 labelled seven-band satellite TIFFs** survive in Git before `50750d2`: 464 Mangrove and 440 NonMangrove. Two inspected files have shape 256×256×7, float32. This is earlier satellite-model data, not the missing RGB drone evaluation set. A recovery manifest is saved without duplicating the large binary files.

The original labelled drone folders and source-level split records were not found in the searched local locations or reachable Git history. To finish the independent evaluation, recover them from the original model developer or the project data custodian, including source/site identities, checkpoint label order and test membership. No one has been contacted automatically.
