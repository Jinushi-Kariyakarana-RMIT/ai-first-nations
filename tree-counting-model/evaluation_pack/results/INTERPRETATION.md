# What this evaluation shows

The frozen current model transfers poorly to this external forest-image
benchmark at the application's current settings. On the 194 original images,
there are 6,633 reference crowns, 94 predictions, and 42 one-to-one matches at
IoU 0.50. Counting MAE is 33.71 crowns per image. A prediction is not necessarily
a correct tree: 52 predictions did not match a reference crown at that threshold.

This is evidence about these particular images and inference settings, not a
statement that the model has the same error on Mourilyan imagery. Differences
in forest appearance, pixel scale and annotation conventions are possible
contributors; this evaluation does not isolate their causal effects.

## Control check

The same unchanged app inference path was run on the nine existing project test
images. It produced 572 predictions against 759 reference crowns, with MAE
20.78. Per-image evidence is in `existing_benchmark_control.json`. The model
therefore does detect many trees on familiar-source imagery; the NEON result
is not simply a failed model load or an empty inference response.

The older notebook's MAE of 15.56 used direct file-path predictions on MPS,
the library's default NMS setting, and post-filtered scores. The app path uses
explicit NMS IoU 0.45, PIL RGB tiles and global duplicate suppression. The two
results are not identical evaluation procedures. This pack deliberately tests
the application's current path; it does not overwrite the historical result.

## What is ready

All 1,164 clean images have labels and expected counts. All originals have
numbered label previews. Label bounds, checksums and counts were validated;
geometric transformation and one-to-one matching tests passed. Published labels
were used without generating pseudo-ground-truth from the evaluated model.

Use this pack to reproduce and investigate the failure and compare future
frozen candidate models. If these data influence training or threshold tuning,
reserve a new untouched benchmark for the eventual final claim. Obtain reviewed
Mourilyan annotations before reporting site-specific accuracy.
