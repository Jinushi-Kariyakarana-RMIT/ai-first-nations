# Current data availability — 14 October 2026

Mourilyan Harbour photos are now external manual-demo inputs in
`DS Proejct/Mourilyan_Harbour_Fresh_Test_Images/`. They are not internal testing,
training or annotation candidates. The old 527-image inventories, review records
and saved notebook outputs are historical snapshots, retained without relabelling
or changing their numerical results.

- Internal tree datasets still resolve through their compatibility shortcuts.
- The tree HTTP checker now uses internal benchmark photos and composites only.
- Benchmark preparation no longer inventories the external demo photos.
- The review UI excludes missing historical sources but preserves their records.
- Classifier refresh requires approved images in `data/testing/classifier_sources/`
  inside the ML workspace. This folder is not currently populated; refresh stops
  before replacing saved evidence. A scored independent benchmark also needs
  reviewed labels, site separation and the documented provenance checks.
- Import an explicit internal benchmark manifest for new annotation work. The old
  `prepare` shortcut cannot reuse the historical Mourilyan inventory.

See [the full project structure](../../docs/PROJECT_STRUCTURE.md). Rerunning the
lightweight overview is safe: it reads saved evidence, not the demo photo folder.
