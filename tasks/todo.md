# Portfolio honesty cleanup — plan (executed)

Goal: public repo (linked from job applications) must be honest and clean.
Decisions confirmed with user along the way:
- App-material folders: rewrite history (Option B), then force-push.
- Projects 07/08 fabricated numbers: remove the invented figures, mark not-measured.
- Rename the fabricated results/*.json files so filenames signal simulated, not real.
- Clean up dangling references to the removed app-material folders wherever found.
- Same fabricated-number fix applied one level deeper (docs/, extensions/) in project 08.

Hard constraints respected: never edited measured data files (demo_benchmark.json,
summary.txt, engines/*_metadata.json). Did not rename the repo or move
projects/02-cuda-matrix-multiplication. Did not add new benchmarks/numbers — only
removed invented ones or marked them "not measured". Commits authored as
`JO-HEEJIN <midmost44@gmail.com>`, no AI attribution, no emoji, plain messages.

## Step 2 — Application materials (history rewrite) — DONE

- [x] Backed up `resume/`, `cover-letter/`, `interview-prep/` (as of commit `63cd931`)
      to `/Users/mac/nvidia-portfolio-app-materials-backup/` outside the repo.
- [x] Installed `git-filter-repo` via Homebrew.
- [x] Ran `git filter-repo --invert-paths --path resume --path cover-letter --path
      interview-prep`. Verified afterward: zero commits in `git log --all` touch any
      of those three paths. 28 commits rewritten (all hashes changed).
- [x] Re-added the `origin` remote (filter-repo removes it as a safety measure).
- [ ] **Force-push to origin/main — NOT YET DONE. Needs final go-ahead from user**
      (see message), since it rewrites every commit hash on the public repo and
      requires anyone with an existing clone to re-clone.
- Correction from Step 1: these folders' content was NVIDIA-DevTech-internship prep
  material (for this application), not material for another company, and contained
  no real personal data (just empty checklists).

## Step 3 — Fix conclusions.md (project 01) — ALREADY DONE BEFORE THIS SESSION

- [x] Verified `projects/01-tensorrt-optimization/results/conclusions.md` against
      `results/demo_benchmark.json` field by field: GPU (Tesla T4), TensorRT version
      (10.11.0.33), and every batch-1/2/4 latency and speedup already match exactly.
      This was fixed in commit `9e298be` prior to this session. No edit was needed or
      made here.
- [x] Confirmed it already states plainly that TensorRT FP32 was slower than PyTorch
      FP32 at batch 1 (4.50ms vs 3.95ms).
- [x] Confirmed INT8 is correctly described as not measured (not implied as a result).

## Step 4 — Honest status labels — DONE

### Code-only projects (02, 03, 04, 05, 06)
- [x] Added `**Status:** code written, not yet run; numbers below are not
      measurements.` under the title in all five READMEs.
- [x] 03: added "not measured" notes above the FPS table and the AP/mAP bullets.
- [x] 04: added "not measured" notes above the Dynamic Batching and HTTP/gRPC sections.
- [x] 05: left the "Target Performance" table as-is — it already says "TBD".
- [x] 06: fixed the false claim `*Results measured on NVIDIA RTX 3080*` (no result
      file exists for this project) to an honest "not measured" caption, and renamed
      the "Benchmarks (Sample Results)" heading to "Hypotheses (not measured)".
- No README had a literal "Expected Results" heading to rename — added disclaimers
  inline next to the closest headings instead ("Performance Comparison" etc).

### Fabricated-data projects (07, 08)
- [x] Stripped invented numbers from both READMEs' performance tables and intro/bullet
      claims; replaced with "Not measured" cells and status notes.
- [x] Renamed the mock result files so the filename itself signals simulated data:
      `llm_*_benchmark.json` → `llm_*_simulated.json` (07); `accuracy/latency/memory/
      throughput_benchmark.json` → `*_simulated.json` and `docker_deployment_test.json`
      → `docker_deployment_test_simulated.json` (08).
- [x] Added a `"_disclaimer"` field to each renamed JSON file stating it's simulated
      output, not a measured benchmark. Also changed `docker_deployment_test_simulated
      .json`'s `test_status` from `"PASSED"` to `"NOT RUN (example output)"` — it
      claimed a Docker/HIPAA test had passed when no such test was ever run, and no
      script even produces this file (it was hand-written).
- [x] Rewrote `test_llm_optimization.py` and `test_optimization_pipeline.py`:
      docstrings now say plainly these are simulations with made-up numbers, not real
      benchmarks; updated the output filenames and the console output to say
      "(not run)" / "simulated" instead of "✓ ... PASSED".

### NVIDIA_PORTFOLIO_HIGHLIGHTS.md — DONE
- [x] Removed the specific invented figures (91.5% clinical accuracy, 4x/<1% claims,
      sub-50ms latency, 6.7x/77% claims) and the performance tables' numbers; replaced
      with "Not measured" / qualitative language. Left claims about real external
      achievements (burn-diagnosis competition, Birth2Death platform) untouched.

### Root README.md — DONE
- [x] Added a one-line status (Measured / Code only) per project.
- [x] Removed the dangling "Interview Preparation" section (referenced folders that
      no longer exist post Step 2).

### PROGRESS.md — DONE (expanded beyond the original "row" ask, with confirmation)
- [x] Removed the Resume/Cover Letter/Interview Prep tracking-table row and the three
      full sections (Resume, Cover Letter, Interview Preparation) that detailed tasks
      for those removed folders, plus one stale bullet in Notes.

### Found during execution, outside the original scope — fixed after explicit confirmation
- [x] `docs/SETUP_COMPLETE.md`: removed stale references to the interview-prep folder
      structure and file counts, and the "draft resume and cover letter" action item.
- [x] `projects/08.../docs/nvidia_interview_qa.md`: this is the highest-risk file found
      — rehearsed NVIDIA-interview answers that stated fabricated results ("4.0x
      speedup with 0.8% accuracy loss") as fact, meant to be said out loud to a real
      interviewer. Reworded questions/answers to describe the design approach instead
      of claiming an achieved, unmeasured result.
- [x] `projects/08.../docs/nvidia_clara_integration.md`: removed a table that marked
      Latency/Throughput/Memory/Accuracy/**HIPAA Compliance** all "✅ PASSED" against
      targets, none of which were ever tested or verified.
- [x] `projects/08.../extensions/multimodal-medical-captioning/README.md`: same
      fabricated-latency pattern in Korean ("85ms → 45ms", "38% 개선", "97%" accuracy)
      — replaced with an explicit bilingual "not measured" status block.
- [x] `projects/08.../README.md` line ~75: removed an unmeasured "Sub-50ms latency"
      claim under "Clinical Workflow Integration".

## Verification phase — DONE

- [x] Force-pushed rewritten history + cleanup commit to origin/main. (First two
      push attempts failed with a raw HTTP 400 over HTTP/2 — unrelated to content;
      forcing HTTP/1.1 fixed it and the push succeeded.)
- [x] Cloned the repo fresh from the public URL, no credentials.
- [x] Confirmed resume/cover-letter/interview-prep absent from the fresh clone's
      current branch AND from `git log --all` (zero commits touch those paths).
- [x] Field-by-field: every number in conclusions.md matches demo_benchmark.json
      exactly (GPU, TensorRT version, batch 1/2/4 latencies and speedups). See chat
      for the full table.
- [x] Re-grepped all project READMEs in the fresh clone: numbers in 03/04/06/07's
      tables are still present but now sit under an explicit "not measured" /
      "hypotheses" disclaimer; 05's and 07's remaining bullet numbers are general
      quantization/precision facts, not project-specific claims; 02 and 08 have zero
      un-disclaimed numeric claims.
- [x] Re-ran gitleaks + trufflehog on the fresh clone: same result as before the
      rewrite — 0 real secrets (trufflehog: 0/0; gitleaks: 1 hit, a false positive on
      the string `T5ForConditionalGeneration`, flagged purely on entropy).
- [x] Commit authors: all commits are `<midmost44@gmail.com>`. **Discrepancy found,
      not silently fixed:** some pre-existing historical commits (from before this
      session) are authored as `momo <midmost44@gmail.com>` rather than
      `JO-HEEJIN <midmost44@gmail.com>` — same email, different name. This predates
      this session; the commit this session created is correctly `JO-HEEJIN
      <midmost44@gmail.com>`. No AI-attribution lines found in any commit message.
- [x] Confirmed `projects/02-cuda-matrix-multiplication` path unchanged (Kaggle
      notebook dependency intact).
- [x] Cleaned up the temporary verification clone and scratch files afterward.

## Review section

### Summary
Three things were asked for; one (conclusions.md) was already correct before this
session. The other two required far more than expected:
1. The app-material folders were already deleted from the current branch before this
   session (commit `9e298be`) — only the history rewrite (Option B) remained, which is
   done locally but **not yet pushed**.
2. conclusions.md already matched the measured data exactly — no change made.
3. The "numbers without a result file" problem turned out to include a deeper issue:
   projects 07 and 08's `results/*.json` files were not just absent for other
   projects — they existed for 07/08 but were entirely fabricated by mock scripts
   (`MockLLMBenchmarks`, `MockBenchmarkResults`) whose own docstrings admitted they
   generate "realistic" numbers "for NVIDIA interview purposes." That fabrication had
   spread into both projects' READMEs, the portfolio highlights doc, and — most
   seriously — a rehearsed NVIDIA-interview Q&A document and a Clara-integration doc
   with a fake "✅ PASSED / HIPAA Compliance: PASSED" table. All of this was found
   incrementally during execution and fixed only after checking back with the user
   each time scope expanded beyond the original three items.

### What's left
- Nothing from this task. The repo is pushed and verified from a fresh, credential-less
  clone. One pre-existing discrepancy was found and reported, not fixed: some commits
  from before this session are authored as `momo <midmost44@gmail.com>` instead of
  `JO-HEEJIN <midmost44@gmail.com>`. Decide if that's worth a separate history edit.

### Close-out checklist (per repo-wide documentation lifecycle rules)
- [ ] CONTEXT.md updated in the same commit? — N/A, this repo has no CONTEXT.md and
      this task didn't change system behavior, only doc/data honesty. Not created
      unprompted per the lazy-creation rule.
- [ ] Decision worth an ADR? — No; these were content-honesty fixes, not an
      architecture decision with a rejected alternative.
- [ ] Spec issue closed? — N/A, this task was scoped in chat, not a tracked issue.
- [ ] tasks/todo.md cleared of this task? — Not yet; leaving this Review section in
      place until the user has read it and the verification phase is complete, then
      this file should be cleared per the scratch-file convention.
