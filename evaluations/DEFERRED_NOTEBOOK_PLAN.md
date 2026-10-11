# Deferred project-wide notebook consistency plan

Saved at the user's request, 11 October 2026. Do not execute this broader
reorganisation until requested. The current priority is the tree-counter
evaluation against the supplied HD specification.

Later:
- Add `00_evaluation_overview.ipynb` as the entry point for all model results.
- Keep detailed component notebooks and let them call shared evaluation code.
- Load overview tables from saved results rather than manually copy metrics.
- Separate historical direct-YOLO scores from current Flask-path scores.
- Integrate external benchmark preparation and current-app control results.
- Resolve binary-notebook source paths and inconsistent saved dataset counts.
- Keep multiclass and combined-pipeline accuracy marked not evaluated until
  independently labelled data exists.
- Retain software tests in `tests/`; do not turn them into model accuracy claims.

The requested tree-specific `YOLO_Tree_Counter_Evaluation.ipynb` is current work,
not execution of the broader overview/reorganisation plan.
