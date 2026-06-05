# Task Checker Report

This report is generated from `HCI.pdf` TODO markers and the workflow status file.

## TODO items found in HCI.pdf

- Page 2: [TODO: age range, handedness, vision;
informed consent / ethics statement] | context: n-house fromtwo healthy adult volunteers[TODO: age range, handedness, vision; informed consent / ethics statement] using a [TODO: device / amplifier model
- Page 2: [TODO: device /
amplifier model, e.g. “ak-channel research-grade EEG head-
set”] | context: med consent / ethics statement] using a [TODO: device / amplifier model, e.g. “ak-channel research-grade EEG head- set”] following the international 10–20 elect
- Page 2: [TODO: confirm; state any hardware notch/anti-alias
filter] | context: gitized at a sampling rate off s= 250Hz [TODO: confirm; state any hardware notch/anti-alias filter]. Fig. 4 shows the acquisition session i
- Page 3: [TODO] | context: Left-to-right blocks:(1) Fixationcross ([TODO] s, baseline)→(2) Cue— one of{ARROW, LET
- Page 3: [TODO] | context: direction Right/Left/Forward/Backward ([TODO] s)→(3) Intent window— subject sustains
- Page 3: [TODO] | context: nal motor intent while EEG is recorded ([TODO] s)→(4) Rest inter-trial interval ([TODO
- Page 3: [TODO] | context: TODO] s)→(4) Rest inter-trial interval ([TODO] s). The block is repeated for [TODO] tr
- Page 3: [TODO] | context: l ([TODO] s). The block is repeated for [TODO] trials per direction; the three cue mod

## Incomplete workflow phases from workflow_status.md

- **Phase 6** — **IN PROGRESS**: Results generated in [results/](file:///d:/EEG/results/)

## How to use this task checker

1. Run `python task_checker.py` to print undone tasks to the console.
2. Use `python task_checker.py --output task_check_report.md` to write a Markdown report.
3. Update the PDF or workflow files, then rerun the checker to confirm the remaining undone items.
