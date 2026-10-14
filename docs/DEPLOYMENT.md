# Deployment and technical handover

This guide starts after the README's local setup and checkpoint installation.
The existing local application has been verified. The deployment configuration
below is a starting configuration to validate on the recipient's infrastructure,
not a claim that their server, network or concurrent workload has been tested.
Waitress 3.0.2 startup was checked locally with all three models loaded and
`/analyse` and `/about` returning HTTP 200 on an isolated port. The test server
was stopped afterwards; the existing application was not interrupted. A clean
dependency installation on a new machine has not been verified.

## What the receiving team needs

- Repository access and the agreed release branch/commit. The current working
  branch is `feature/mangrove-ai-enhancements`; record the exact commit with
  `git rev-parse HEAD` rather than relying on a moving branch indefinitely.
- The separately supplied tree checkpoint and both classifier checkpoints.
- Python/package availability for the target OS and architecture. The application
  pins reflect the local macOS ARM64/Python 3.14.7 environment. If a package has no
  compatible wheel on your host, establish and test a compatible environment;
  do not assume the local pins prove cross-platform support.
- Writable persistent storage for `machine learning/flask-application/uploads/`.
  Model files can be read-only. Training images are not required for inference.
- A host with memory and processing capacity measured against representative image
  sizes. There is no established hardware minimum or response-time SLA. GPU use
  is optional; do not assume the application selects the same device for every model.

Checkpoint SHA-256 references are recorded in
`../machine learning/evaluations/classifier_hd/environment.json` and
`../machine learning/evaluations/tree_counter_hd/deployment_validation/flask_summary.json`.
Compare supplied model files against those records before calling them the evaluated
checkpoints. A functional startup with a different checkpoint is not evidence of
the same model performance.

## Server startup example

Use a production WSGI server. Flask's official [Waitress deployment guide](https://flask.palletsprojects.com/en/stable/deploying/waitress/)
explains installation, application loading and reverse-proxy placement.

With the virtual environment activated, from the repository root:

```bash
python -m pip install waitress==3.0.2
```

Configure a **stable, private** `FLASK_SECRET_KEY` in the service environment and
set `FLASK_DEBUG=false`. Generate a value once with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Save that value in the deployment platform's secret configuration. Do not commit
it, put it in screenshots or regenerate it on each restart. The application does
not itself load a `.env` file; configure actual environment variables.

Then, in that same activated environment:

```bash
cd "machine learning/flask-application"
waitress-serve --host=127.0.0.1 --port=5001 --threads=1 app:app
```

Stop the local Flask server first if it occupies port 5001. The single-thread
configuration deliberately serialises requests until shared model concurrency is
validated. Long inference requests will delay other requests; this is not a queue
or a demonstrated multi-user capacity limit. Install/run Waitress as the normal
application user, not an administrator/root user.

Place your organisation's reverse proxy in front of this local listener. Configure
HTTPS, organisation login or equivalent access restrictions, and a user-facing
hostname. The application has no built-in authentication or per-user file isolation;
do not expose it directly to the public internet. Keep proxy upload limits aligned
with the application's 50 MB limit and measure timeouts against actual inference.

Use your host's service manager for restart-on-failure, working directory,
environment variables and logs. Record the deployment-specific proxy/service
configuration in your operations repository. These files are not provided here
because the recipient's hosting platform has not been specified.

## Data and operations

- Uploaded photos and overlays are stored on the host, not sent to a remote
  inference API by application code. Apply your organisation's access, retention
  and backup rules to that folder. Review dependency/network policy separately.
- Upload files survive a process restart if the storage persists; the recent-history
  list is only in process memory and resets. Multiple worker processes do not share it.
- There is no automatic deletion, durable job queue, account system or audit-history
  database in the current application. Plan these separately if the deployment needs them.
- Keep model weights with the release and retain the previous version for rollback.
  Update code and its matching weights together, then restart the service.
- Development demo photos and the original logo sit outside Git. They need a separate
  handover only if the recipient wants those assets; the app already has its own assets.

## Acceptance on the recipient's host

1. Confirm all three checkpoints load, without unexpected training or downloads.
2. Open `/analyse` and `/about` through the intended user-facing address.
3. Upload an approved representative JPG/PNG. Check predictions, image dimensions,
   overlay access and any partial-failure warning.
4. Check corrupt/oversized uploads and two uploads with the same original filename.
5. Measure typical and largest expected images for memory and response time; validate
   expected simultaneous users before increasing workers/threads.
6. Restart the service and verify it recovers with the configured model/storage paths.

These checks prove operational behaviour. Independent labelled evaluation is still
required to establish local survey accuracy.

## Technical verification

From the repository root, with the environment activated:

```bash
python -m pip install pandas scipy scikit-image psutil requests beautifulsoup4 pycocotools nbformat nbclient nbconvert ipykernel
python -m unittest discover -s "machine learning/tests" -v
python -m compileall -q "machine learning/flask-application" "machine learning/tree-counting-model" "machine learning/tests"
```

The test dependencies supplement `requirements-app.txt`. Saved component environment
records describe previous evaluation runs, not a complete portable environment lock.
Do not run every historical notebook as a startup check; some contain training code.
The lightweight overview can be refreshed without rerunning model evaluation:

```bash
python "machine learning/evaluations/overview/build.py"
```

Evaluation dataset shortcuts are relative symbolic links. Preserve symlinks when
copying the workspace; Windows Git setups may require symlink support for evaluation
work. The application inference path does not require those dataset shortcuts.

## Troubleshooting

| Symptom | Action |
|---|---|
| Tree counting unavailable | Supply the evaluated `best_tree_counter.pt` at the README path; restart and inspect the loading error |
| Classifier unavailable | Check classifier file presence, hashes and compatible Torch/Torchvision packages |
| `TrainingData` missing message | Existing classifier inference can still load; do not start training merely to suppress this message |
| Package cannot install | Check target Python/OS/architecture compatibility; the recorded local environment is not a universal lock |
| Port 5001 already in use | Stop the previous project server or choose another port; update the proxy/user URL accordingly |
| A colleague cannot open localhost | Localhost refers to their own computer; use the configured server hostname |
| Partial-failure warning | Inspect the affected result card and server log; other model outputs may still be available |
| Upload rejected | Use a readable JPG/JPEG/PNG below 50 MB; very large decoded images may also be rejected |
| Out of memory or slow response | Measure a smaller representative image and host resources; do not increase concurrency without testing |
| Classifier evaluation refresh stops | Approved internal sources are absent or differ from the saved manifest; follow the evaluation workflow, not the external demo-photo folder |
