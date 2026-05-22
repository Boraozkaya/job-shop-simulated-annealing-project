# Job Shop Scheduling with Simulated Annealing

This project implements a Job Shop Scheduling Problem (JSSP) solver from scratch
for the ENM404 project assignment. The selected metaheuristic is Simulated
Annealing (SA), and the objective is to minimize makespan (`Cmax`).

All tables, charts, and presentation visuals in this package are generated from
the included Python code and the benchmark files under `data/raw/`.

## Folder Structure

- `src/`: benchmark loader, JSSP decoder, and SA algorithm
- `data/raw/`: selected JSPLIB / OR-Library benchmark files
- `outputs/tables/`: experiment CSV files and metadata
- `outputs/figures/`: charts and Gantt visuals used in the presentation
- `presentation/`: final PowerPoint file
- `docs/`: project brief and written project summary

## Quick Start

For live presentation, the easiest option is:

- Mac: double-click `RUN_DEMO.command`
- Windows: double-click `RUN_DEMO_WINDOWS.bat`

```bash
python run_all.py
python demo_run.py --instance ft10
python build_presentation_assets.py
```

For a short live demo during the presentation, use:

```bash
python demo_run.py --instance ft10 --iterations 2500
```

`run_experiments.py` regenerates the full experiment set. It can take longer
than the demo because it performs parameter tuning and 10 independent runs per
benchmark instance.

The final PowerPoint is already generated under `presentation/`. The optional
`build_presentation.cjs` script was used to build the deck from the generated
figures.

## Selected Benchmark Instances

- `ft06` (6x6): Easy
- `la01` (10x5): Moderate
- `ft10` (10x10): Difficult
- `la16` (10x10): Difficult
- `la21` (15x10): Very Difficult

Benchmark files come from JSPLIB / OR-Library references and are included
locally so the project can be run without downloading data again.
