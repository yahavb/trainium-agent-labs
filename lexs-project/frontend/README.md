# NKIEvolve frontend

The three-tab UI is plain HTML, CSS and JavaScript. No Node installation or build
step is needed. Keep this `frontend/` directory next to `nki-kernel-gen/` on the
Trainium host. Start `python backend.py` there, then open its printed ngrok URL.
Paste the **API access token** printed by the script into the connection bar.
OpenAI, W&B and ngrok credentials stay in the worker environment.

- **Launcher:** Paste a reference module, or describe the operation and generate
  an editable PyTorch reference using GPT-6 Luna. Review and verify it, choose
  Luna/Sol/Astra, set the search budget, then start evolution. An optional initial
  NKI kernel can be supplied; otherwise the existing runner bootstraps one.
- **Logs:** Live speedup and latency charts, correctness results, console output,
  W&B link and a stop button. Select previous runs from the run menu.
- **Kernel Code:** Best measured correct NKI source, with copy and download.

The API token is stored in this browser tab's session storage so refreshing can
reconnect. Disconnect clears it and leaves any worker job running. Reference
drafts are not persisted. Code or description changes clear verification.
Each generation/launch uses paid model calls. Description generation does not
execute the reference or start evolution. The user must explicitly approve the
code before launch. The backend checks syntax and required entry points; it
does not sandbox uploaded Python.

For local UI development, start `python backend.py --local-only` from
`nki-kernel-gen/` and open `http://127.0.0.1:8000`. Hardware search still needs
Trainium; the frontend has no fabricated metrics or automatic paid requests.
It follows the same local metrics that the runner sends to W&B, polling every
2.5 seconds while a run is active. Baseline is iteration 0; failed candidates
are included in the table but excluded from performance charts. Speedup is
relative to the initial NKI kernel. The lowest latency and best geometric speedup
can correspond to different candidates.

Optional frontend logic tests (development only): `npm install`, then `npm test`
from this directory. These use a simulated API and do not make paid requests.
