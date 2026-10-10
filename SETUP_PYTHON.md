# Python environment for the kernel agent

**On your seat you don't need any of this.** The seat already has Python 3.13, `nki` 0.6.0 and `neuronx-cc`.
Set this up only to grade NKI kernels on your own machine.

**Works on Linux x86-64, or Windows with WSL2.** `nki` has no Mac version; on a Mac, use Docker (step 3).

## 1. The NKI simulator

This is enough to grade kernels with `nkibench.py`:

```bash
python3.12 -m venv ~/venvs/nki
~/venvs/nki/bin/pip install --extra-index-url https://pip.repos.neuron.amazonaws.com \
    "nki==0.6.0" "numpy==2.5.3" "httpx==0.28.1"
```

Check it:

```bash
cd trainium-agent-labs/projects/02-kernel-agent
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2      # always set this, or nki simulates the wrong chip (trn3)
~/venvs/nki/bin/python nkibench.py --selftest     # expect: SELFTEST PASSED
```

## 2. Optional: the trn2 compiler

This checks that a kernel really compiles for the chip. Linux x86-64 only; it doesn't run on a Mac, even
in Docker.

```bash
python3.12 -m venv ~/venvs/cc
~/venvs/cc/bin/pip install --extra-index-url https://pip.repos.neuron.amazonaws.com \
    "neuronx-cc==2.27.5334.0" "nki==0.6.0" "numpy==2.5.3"
~/venvs/cc/bin/neuronx-cc --version
```

When you use it, put `~/venvs/cc/bin` first on your `PATH`; nki looks for the compiler there.

## 3. On a Mac

Run step 1 inside a Linux container:

```bash
cd trainium-agent-labs/projects/02-kernel-agent
docker run --rm -it --platform linux/amd64 -v "$PWD":/repo -w /repo python:3.12-slim bash
# then, inside the container, step 1's two commands, then:
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 ~/venvs/nki/bin/python nkibench.py --selftest
```

## Tips

- Python 3.12 and 3.13 both work.
- Don't run Python from a folder that contains a subfolder named `nki`. It hides the real package, and
  `import nki` silently picks up the wrong one.

## How this was checked

- **Step 1** was tested in a clean `python:3.12-slim` container (linux/amd64). It installed
  nki 0.6.0+31049202112.g85070674, the same build as on the seats, and `nkibench.py --selftest` passed.
- **Step 2's versions** are the ones the team's Linux test machine used to compile kernels for trn2 and run
  them in birsim.
