# NKIEvolve — NKI kernel evolution on Trainium

Provide a trusted PyTorch reference, evolve NKI candidates with OpenEvolve, and
inspect correctness, hardware latency and the best measured kernel in a browser.

- [Backend and setup](nki-kernel-gen/README.md)
- [Frontend](frontend/README.md)
- [Softmax run evidence](docs/evidence/README.md)

Install the Neuron SDK on a Trainium host, then from `nki-kernel-gen/` run
`python -m pip install -e '.[server,tracking]'` and `python backend.py`.
Set provider and tunnel credentials in the worker environment as described in
the backend README. Open the printed endpoint and enter its API access token.
