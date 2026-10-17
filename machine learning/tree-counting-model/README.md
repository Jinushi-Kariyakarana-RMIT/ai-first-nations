# Tree counter

`tree_counter.py` provides tiled YOLO inference and removes overlapping detections before counting visible tree crowns. `best_tree_counter.pt` is the saved model used by the Flask application.

Run the application using the [main guide](../../README.md). No training dataset is needed for inference. Historical training/evaluation utilities and datasets are in the separate research archive; the training entry point in this module is not part of application startup.
