# Project decisions (agreed with the owner, 2026-10-28)

- **Disk budget:** at most 25 GB for the project. Raw video is streamed:
  downloaded, tracked, then deleted.
- **Licence target:** non-commercial research release. Dataset provenance is
  kept per clip, so the model can be retrained on commercial-friendly data
  only (GRID CC BY 4.0, CREMA-D ODbL) by dropping RAVDESS.
- **Compute:** laptop only (Apple M5, 16 GB). No cloud spend.
- **Sign-ups:** none. Only openly downloadable datasets are used.
- **Language:** English first.
- **Repo:** local git only. Nothing is pushed or published.
- **"STT speed"** is read as live microphone audio → face in real time, with
  under ~150 ms of added delay.
