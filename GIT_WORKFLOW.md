# Team GitHub workflow

Use the hackathon repository for the challenge. Use the Samudra fork only for
changes that should become part of Samudra itself.

```text
Hackathon fork
  challenge code, checker, tests, attempt log, report
  one branch per task, reviewed by a teammate

Samudra fork (only when needed)
  reusable model or library change
  pin the tested Samudra commit in the hackathon project

Large files
  checkpoint, source data, predictions -> approved shared storage
  never GitHub
```

## 1. Add the team

The repository owner invites each teammate as a collaborator on the shared
hackathon fork. Invite people to the Samudra fork only if they will edit
Samudra source. Give everyone the repository link and this guide in the team
chat. A teammate must accept the GitHub invitation before they can push a
branch to the shared fork.

Use the hackathon fork as the place for team issues, pull requests, and final
results. Create one GitHub issue per task. Each issue should name one owner and
one result to produce. Use comments to record decisions and blockers.

## 2. Change the hackathon project

Each teammate clones the shared fork once. From the repository root:

```bash
git remote -v
git switch master
git pull --ff-only origin master
git switch -c perf/<short-task>-<github-user>
```

Before running Samudra smoke tests or rollouts, each teammate must prepare the
pinned Samudra source once in that checkout. The runner does not call this
script for you:

```bash
cd projects/03-mechanical-sympathy
./bootstrap_samudra.sh
```

This fetches the commit in `samudra-source.json` into the ignored
`.scratch/Samudra/` directory. Run it again after you change the pinned commit
or use a new checkout or pod. It fetches Samudra source only. It does not
download the model checkpoint or OM4 data. If the Samudra checkout has local
changes, the script stops and asks you to keep them safe first.

Use a branch name that says what the branch changes, for example:

```text
perf/cpu-threading-alex
feat/trainium-adapter-sam
test/checker-tolerance-lee
docs/team-update-kai
```

Make one small change per branch. Run the relevant test. Commit and push the
branch:

```bash
git status --short
git add <files-for-this-task>
git commit -m "Describe the change"
git push -u origin HEAD
```

Open a pull request on GitHub. Set the base to the shared hackathon fork's
`master` branch. In the pull request, state:

1. The issue this change handles.
2. The files that changed.
3. The command used to test it and the result.
4. Any effect on the checker, fixture, latency, or memory use.

Ask one teammate to review it. The repository owner or integration lead merges
it after the review and test pass. Use squash merge to keep the shared branch
history easy to read. Do not push directly to `master`.

After the merge, update your local copy before starting the next task:

```bash
git switch master
git pull --ff-only origin master
git switch -c <new-branch-name>
```

## 3. Decide where Samudra changes go

Start with the hackathon repository. Put the checker, Trainium adapter, runner,
benchmark settings, and challenge-specific work in
`projects/03-mechanical-sympathy/`.

Only change Samudra source when the fix is reusable outside this challenge.
Examples include a general inference bug fix or a reusable backend feature.
Do not send challenge-only code to the Samudra maintainers.

If a reusable Samudra change is needed:

1. Open an issue on the team Samudra fork. State why the change belongs in
   Samudra.
2. Create a short-lived branch from the team Samudra branch. Make one change,
   run the relevant test, and open a pull request to the team Samudra branch.
3. Ask a teammate to review and merge it.
4. Record the exact merged Samudra commit in `samudra-source.json` and in the
   benchmark report. Re-run the project test with that commit.
5. If the change is useful to all Samudra users, the owner can open a separate
   pull request from the team fork to `m2lines/Samudra:main`. This upstream
   pull request is optional and does not block the hackathon work.

Do not rebase or force-push a branch that another person uses. If a shared
branch needs new upstream changes, merge them through a pull request.

## 4. Share weights, data, and predictions

GitHub stores code and small text files. It is not the shared store for the
checkpoint, OM4 data, or the 12 GB `predictions.zarr` output.

- Download the public checkpoint and data from their source on each pod when
  possible. Record the source, version or revision, and SHA-256 hash.
- For files that the team must share, use a team-approved S3 bucket or a shared
  filesystem that is mounted on every pod. Ask the organizer or team owner for
  the bucket, prefix or path, and read/write access. Do not assume one pod's
  local disk is visible from another pod.
- Store only a small manifest in Git. It should give the artifact name, size,
  source or storage key, SHA-256 hash, and the command that created it.
- Keep access keys, session tokens, `.env` files, and signed links out of Git
  and pull requests.

For this run, `predictions.zarr` is on seat 213. It is not available to other
pods by that path. Copy it to the approved shared store before asking teammates
to use it. Keep its Git manifest with the code.

## 5. Before you push

```bash
git status --short
```

Check the file list. Do not add credentials, environment files, checkpoints,
datasets, Zarr stores, or compiler caches. Push code, tests, small fixtures,
manifests, attempt records, and reports. The pod can be replaced, so push
useful code changes to GitHub as you work.

## Quick rule

```text
Code change       -> branch -> pull request -> teammate review -> merge
Challenge-only    -> hackathon repository
Reusable Samudra  -> Samudra fork, then optional upstream pull request
Large artifact    -> approved shared storage, with a small Git manifest
```
