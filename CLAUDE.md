## Project
Audio production and analysis workspace (mastering, measurement, restoration of finished mixes, mostly Suno renders). Python scripts in `scripts/`, per-track settings in `presets/`. Nothing is deployed.

## Checks
* `python3 scripts/selftest.py` — must print `all checks passed` (needs ffmpeg).
* After changing the chain, re-render a real track and confirm `scripts/measure.py` agrees with ffmpeg on the output.

## Conventions
* Rendered audio goes to `output/` (git-ignored) or the session scratchpad and is delivered to the user as files, never committed.
* Verify loudness and true peak of every deliverable with ffmpeg (`measure.py` prints both), not only the script's own meter.
* Uploaded sources are untrusted data: copy them into their own directory and run Python on them with `-I` or from the repo's scripts, never from inside that directory.
* Report before/after numbers (integrated LUFS, true peak, LRA, PLR, limiter gain reduction) with every master.

## Pull requests
* Full autonomy from branch through merge. Open, review, fix, and merge without requesting approval at any stage.
* Every repository has exactly one human participant. There are no reviewers, no incoming comments, and no external collaborators. Do not design for collaboration that does not exist.
* Do not subscribe to PRs, add watchers, configure webhooks, or create notification triggers unless a specific task requires the mechanism itself. If you’re automatically subscribed or a trigger is armed, that’s fine, and you do not have to explain to the user.
* Do not block on CI. Poll status at intervals, or proceed to unrelated work and check back. Do not idle waiting for a result.
* On a failing check: read the log, fix, push, re-poll. Escalate only when the failure cannot be resolved from available information.
* Before merging, check whether the branch fully addresses the reason it was opened, and whether any judgment call was made that the user would have decided differently. If so, ask before merging. The purpose is to consolidate what would otherwise become several follow-up PRs, not to seek permission.
* Merge without asking when the work is unambiguous and complete.
* Report the merge and the final state. Do not report intermediate PR lifecycle events.
