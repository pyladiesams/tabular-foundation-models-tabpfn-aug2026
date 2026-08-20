# Getting Started with Tabular Foundation Models with TabPFN
### Presentation: [{Getting Started with Tabular Foundation Models with TabPFN}](workshop/getting-started-with-tfms.pdf)

| | |
|---|---|
| Participant notebook | [`workshop/tabpfn_workshop.ipynb`](workshop/tabpfn_workshop.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TuanaCelik/tabular-foundation-models-tabpfn-aug2026/blob/main/workshop/tabpfn_workshop.ipynb) |
| Solutions | [`solutions/tabpfn_workshop_solutions.ipynb`](solutions/tabpfn_workshop_solutions.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TuanaCelik/tabular-foundation-models-tabpfn-aug2026/blob/main/solutions/tabpfn_workshop_solutions.ipynb) |

## Workshop description

During this workshop, you'll get an intro into the new era of tabular foundation models
(TFMs), how they differ from traditional ML models, as well as the new capabilities they
unlock for AI agents. We will start with a light intro into TFMs, how they operate and
what their capabilities are. Following this, we will do some hands on exercises with
TabPFN for regression, classification and interpretability, as well as some more advanced
tasks like time-series predictions.

If you have ever prompted an LLM with a handful of examples and watched it pick up the
pattern, you already know **in-context learning**. TabPFN applies the same idea to
tables: you hand it labelled rows as *context* and it predicts new rows in a single
forward pass — no gradient descent on your data, no hyperparameter search.

No prior knowledge of tabular foundation models is assumed. If you know
`scikit-learn`'s `fit()` / `predict()` and have seen a few-shot prompt, you have
everything you need. We use **TabPFN-3** through the Prior Labs API, so **no GPU
required**.

### Session outline

| | Section | What happens |
|---|---|---|
| 0 | Setup | API key, install, imports |
| | In-context learning for tables | 50 rows of context, one forward pass, ~0.99 ROC AUC |
| 1 | **Classification** | Malignant vs. benign, calibrated probabilities, the ICL scaling curve |
| 2 | **Regression** | Same two lines of code, but the answer is a number |
| 3 | **Interpretability** | SHAP beeswarm, per-patient waterfall, partial dependence |
| 4 | Forecasting | 24-month forecast in three lines, via TabPFN-TS |
| 5 | Advanced | Thinking mode: spend fit-time compute, optimise your own metric |
| | Predictive agents | TabPFN as a tool an LLM agent can call (MCP) |

Datasets, all open — nothing to download by hand, nothing sensitive leaves your machine
beyond what is already public:

- [**Breast cancer Wisconsin**](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)
  (ships with scikit-learn) — classification, section 1
- [**Diabetes**](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
  (ships with scikit-learn) — regression, section 2
- **Heart disease / statlog** (fetched from OpenML) — interpretability, section 3. Chosen
  because half its columns are categorical and its values are human-readable, which makes
  the explanations mean something: *"chest pain type = asymptomatic raises risk by 13
  points"*.
- **Mauna Loa CO₂** (ships with `statsmodels`) — forecasting, section 4

## Requirements

* PyLadies Amsterdam uses [uv](https://docs.astral.sh/uv/) for dependency management
* Google account if you want to use [Google Colab](https://colab.research.google.com/)
* A **free Prior Labs API key** →
  [sign up here](https://ux.priorlabs.ai/?utm_source=workshop&utm_campaign=pyladies),
  confirm your email, then copy the key from the **API Keys** page. Please do this
  *before* the session — the email confirmation is the one step that can cost you five
  minutes of the workshop.
* Python ≥ 3.10 (required by `tabpfn-client`) if you run locally.

Section 0 of the notebook offers **two ways to run TabPFN**, and you pick one:

* **via the client** (what we use on the day) — inference happens on Prior Labs'
  servers, so **no GPU needed** and nothing to download beyond the small client package.
* **locally**, with the open-source [`tabpfn`](https://github.com/PriorLabs/TabPFN)
  package — your data never leaves your machine, but you want a GPU for it to be quick,
  and you have to accept the model licence on first use. Sections 1–4 work either way;
  thinking mode in section 5 is API-only.

## Usage

### with Google Colab

1. Visit [Google Colab](https://colab.research.google.com/)
2. In the top left corner select "File" &#8594; "Open Notebook"
3. Under "GitHub", enter the URL of the repo of this workshop
4. Select `workshop/tabpfn_workshop.ipynb` (or the solutions notebook)
5. Run the first cell — it installs everything the notebook needs

Or just click the Colab badge at the top of this README. You do **not** need to clone the
repo in Colab: the notebook installs its own dependencies and every dataset is fetched
from `scikit-learn`, OpenML or `statsmodels`.

### with uv

Run the following code:

```bash
git clone https://github.com/TuanaCelik/tabular-foundation-models-tabpfn-aug2026.git
cd tabular-foundation-models-tabpfn-aug2026

# create and activate venv, install dependencies
uv sync
```

`uv sync` installs everything, including the section 3 and 4 packages — which bring the
open-source `tabpfn` package and torch with them, so the first sync is a large download.

**Picking the kernel.** `uv sync` creates `.venv/` in the repo, and `ipykernel` is
included, so it works as a notebook kernel directly:

- **VS Code / Cursor:** open the notebook, click the kernel picker (top right) →
  *Python Environments…* → the `.venv` entry for this folder.
- **JupyterLab** started with `uv run jupyter lab`: it is already running inside the venv,
  so the default *Python 3 (ipykernel)* kernel is the right one.
- **A Jupyter you started elsewhere:** register the venv once, then pick it by name:
  ```bash
  uv run python -m ipykernel install --user --name tabpfn-workshop --display-name "TabPFN workshop"
  ```

### with pip

```bash
pip install -r requirements.txt
```

### for a workshop giver

The required Python version is set in `pyproject.toml` (`requires-python = ">=3.10"`,
because that is what `tabpfn-client` needs). Add packages with `uv add <package>`.

`requirements.txt` is **hand-maintained and deliberately unpinned**, rather than
generated with `uv export`. A fully pinned export is ~290 lines and would force Colab to
reinstall numpy, pandas and friends mid-workshop, triggering a runtime restart. `uv.lock`
is still there for reproducible local installs.

**The two notebooks.** `solutions/tabpfn_workshop_solutions.ipynb` holds the complete
code; `workshop/tabpfn_workshop.ipynb` is the same notebook with the exercises blanked
out. In the solutions notebook, every answer sits inside a marked block:

```python
# --- YOUR TURN ------------------------------------------------------------
# TODO: what the participant has to do
# --- SOLUTION ---
the answer
# --- END SOLUTION ---
```

and in the participant notebook those blocks are replaced by `# your code here`. If you
change one notebook, **make the matching change in the other by hand** — and clear the
outputs of the participant notebook before committing.

**Notes for running it live:**

- **API budget.** Sections 0–2 are cheap (about a dozen calls). **Section 3 is the
  expensive one**: the SHAP explainer and the partial dependence plots work by predicting
  many times against one fitted context. It passes `fit_mode="fit_with_cache"` so the
  server reuses the encoded context, which is what keeps it to a couple of minutes rather
  than much longer.
- **Installs happen once, in section 0** — one cell for the local route, one for the
  client route, and people run whichever they picked. Both pull in
  `tabpfn-extensions[interpretability]`, `shap` and `tabpfn-time-series` (needed by
  sections 3 and 4 whichever backend you use), which brings torch along, so it takes a
  minute or two. Get the room to run that cell first and talk over it: nobody then hits a
  missing module mid-section, and any Colab restart prompt happens before you start.
- **The client is the default.** In section 0's import cell the client line is active and
  the local one is commented out, matching "this is our preference for today". If you
  would rather demo local, swap which line is commented — **in both notebooks**.
- **Thinking mode (section 5) is off by default.** Thinking fits draw on a separate
  monthly quota (20 by default), so treat it as a presenter demo, not a room-wide
  exercise.
- **Timing:** sections 0–3 are the core (~50 min). Sections 4 and 5 are the parts to
  compress if Q&A is lively. Predictive agents is a 2-minute teaser for the follow-up
  workshop, not an exercise.
- **Keep the UTM parameters** (`utm_source=workshop&utm_campaign=pyladies`) on the Prior
  Labs links — that is how this cohort is attributed.
- **Colab badges** point at `TuanaCelik/tabular-foundation-models-tabpfn-aug2026`. If the
  canonical repo ends up under `pyladiesams`, update the two URLs at the top of this
  README and the badge inside each notebook's first cell.

## Video record

Re-watch [this YouTube stream](https://www.youtube.com/live/PY7ecCQfOv0?si=kVXmevHEJ_aubCpq)

## Credits

This workshop was set up by @pyladiesams and @TuanaCelik

## Appendix

### Further reading

- [Prior Labs docs](https://docs.priorlabs.ai/?utm_source=workshop&utm_campaign=pyladies)
  — quickstart, cookbook, thinking mode, forecasting, MCP
- [TabPFN-3 technical report](https://arxiv.org/abs/2605.13986)
- [`tabpfn-client`](https://github.com/PriorLabs/tabpfn-client) (API) and
  [`tabpfn`](https://github.com/PriorLabs/TabPFN) (open source, runs on your own GPU)
  — local + client backends, XGBoost/CatBoost baselines, SHAP

### Pre-Commit Hooks

To ensure our code looks beautiful, PyLadies uses pre-commit hooks. You can enable them by
running `pre-commit install`. You may have to install `pre-commit` first, using `uv sync`,
`uv pip install pre-commit` or `pip install pre-commit`.

Happy Coding :)
