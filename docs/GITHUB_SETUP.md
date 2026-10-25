# GitHub bootstrapping checklist — by A

1. Install Git, Python 3.10+ (recommend 3.11), VS Code, GitHub CLI (`gh`). Authenticate: `gh auth login`.
2. Extract ZIP, open its root in VS Code; check `git status`.
3. Create a **private** GitHub repo using GitHub website (without README/license because local repo already includes README), suggested `adverse-weather-segmentation`. Or with CLI: `gh repo create adverse-weather-segmentation --private --source . --remote origin --push` after a first commit.
4. `git init -b main`, `git add .`, `git commit -m "chore: initialize research repository"`; then `gh repo create adverse-weather-segmentation --private --source . --remote origin --push`.
5. Add Person B via GitHub Settings > Collaborators. Configure Rulesets / Branch protection for `main`: pull request required, 1 approval, status check `test`, prohibit force push, no direct pushes. On a 2-person private repo, check your plan supports chosen rules; if not, enforce process manually until available.
6. Both clone repo and work only in dedicated feature branches. A approves all PRs; B reviews A's PRs as additional peer review.

## First tickets
- #1 A: initial repo + access + CI, due 26 Oct
- #2 B: environment/hardware note, due 28 Oct
- #3 A: run/offline smoke, due 28 Oct
- #4 B: dataset provenance and licensing notes, due 28 Oct
- #5 B: official 19-class label converter + validation, due 31 Oct
- #6 A: baseline configuration/experimental protocol, due 31 Oct
- #7 B: data audit + split report, due 03 Nov
- #8 A: subset training and mIoU verification, due 03 Nov

Create GitHub Issues manually, assign owner and use dates. Do not call a task 'done' without its exit criteria.
