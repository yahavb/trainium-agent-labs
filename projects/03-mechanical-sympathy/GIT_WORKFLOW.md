# Git workflow

## Repository map

```text
yahavb/trainium-agent-labs:master
              |
              v
pranoyghosh/trainium-agent-labs:master
              |
              v
         team/hackathon
          ^      ^       ^
          |      |       |
       cpu/*  trainium/*  checker/*
              |
              v
pranoyghosh/Samudra:team/trainium
              ^
              |
m2lines/Samudra:main
```

Create both forks on GitHub before you run the commands below. Add each teammate
as a collaborator on both forks.

## Hackathon repository on the pod

Run these commands from `/workspace`:

```bash
git remote rename origin upstream
git remote add origin https://github.com/pranoyghosh/trainium-agent-labs.git
git fetch upstream master
git fetch origin
git switch -c team/hackathon upstream/master
git push -u origin team/hackathon
git switch -c cpu/baseline-pranoy
```

Keep `master` as the upstream mirror. Merge task pull requests into
`team/hackathon`. Use squash merges.

Use these branch forms:

```text
cpu/baseline-pranoy
trainium/<task>-<github-user>
checker/correctness-<github-user>
docs/submission
```

## Samudra repository on the pod

Run these commands from
`/workspace/projects/03-mechanical-sympathy/.scratch/Samudra`:

```bash
git remote rename origin upstream
git remote add origin https://github.com/pranoyghosh/Samudra.git
git fetch upstream main
git fetch origin
git switch -c team/trainium upstream/main
git push -u origin team/trainium
```

Use one `trainium/<task>-<github-user>` branch for each independent task. Merge
reusable changes into `team/trainium`. Update `samudra-source.json` after each
accepted change.

## Synchronize upstream changes

Do not rebase a shared `team/*` branch. Update it through a pull request from a
temporary synchronization branch.

For the final upstream submission, create a new `submit/*` branch and rebase
that branch on the latest official branch. No teammate works directly on a
`submit/*` branch.

## Local editing

Clone the full hackathon fork locally:

```bash
git clone https://github.com/pranoyghosh/trainium-agent-labs.git \
  /Users/pranoyghosh/trainium-agent-labs
nvim /Users/pranoyghosh/trainium-agent-labs/projects/03-mechanical-sympathy
```

Keep `/Users/pranoyghosh/03-mechanical-sympathy` until its files are committed
and compared with the full clone.

## Before each push

```bash
git status --short
python -m unittest discover -s projects/03-mechanical-sympathy/tests -v
```

Confirm that Git does not show checkpoints, data, Zarr stores, compiler caches,
environment files, or credentials.
