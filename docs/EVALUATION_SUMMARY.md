# Evaluation summary

The application implements mangrove detection, Orange/Red/Yellow category classification and YOLO visible-tree-crown estimation. Functional operation is established locally; independent mangrove-survey accuracy and full HD evaluation requirements are not established.

The latest manual trial used nine Mourilyan photos across three drone routes. All completed; median analysis time was about 13.8 seconds on the development machine. These photos have no reviewed reference labels, and training independence is not proven. Confidence percentages are not measured accuracy.

Known findings from that trial:

- Resizing one photo changed the count from 36 to 1 and the predicted category from Orange to Red. Results are sensitive to resolution.
- A GIF renamed with a JPG extension passed upload validation.
- “Download Result” downloads the original photo rather than a result report.
- Crown outlines can be difficult to see against vegetation.

These findings remain open; folder cleanup does not fix them. Counts cover visible crowns of all tree types, and overlapping survey images must not be summed as unique trees. Category names have not been verified as biological species.

Before cleanup, 32 automated tests passed; the manual trial passed 27 of 29 checks and six direct model/response comparisons. Seven application-focused tests are retained in this repository; research tests and all detailed evidence are preserved in the separate local research archive. Tests of software behaviour do not establish predictive accuracy.

Independent reviewed labels, trustworthy train/test separation and representative accuracy evaluation are still needed for full HD claims. The research archive contains the rubric reviews, notebooks and model evaluation limitations.
