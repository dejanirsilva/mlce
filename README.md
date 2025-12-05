# Machine Learning for Computational Economics (MLCE)

This repository hosts the **public course website**, lecture notes, slides, replication code, and Pluto notebooks for the course:

**Machine Learning for Computational Economics**  
Instructor: **Dejanir Silva** (Purdue University)  
Institution: **EDHEC Business School**

📘 **Course website:**  
👉 [https://dejanirsilva.github.io/mlce](https://dejanirsilva.github.io/mlce)

---

## 📂 Repository Contents

### Website Files

- `index.html` – Course homepage (generated via Quarto from `index.qmd`)
- `Module0X/` – HTML slides and figures for each module (e.g. `Module01/Module01_Slides.html`)
- `notebooks/` – Static HTML versions of the interactive Pluto notebooks
- `Lecture_Notes_MLCE.pdf` – Complete lecture notes for the course

### Source Code

- `src/Module0X/` – Julia source code for each module, including:
  - Example scripts and numerical experiments
  - Benchmarking and illustration code
  - DPI implementation routines
  - Pluto notebooks (files of the form `NB_*.jl`)

---

## 🎓 Course Slides

Each module has a full HTML slide deck (also linked from the course website):

- **Module 01 – Introduction**  
  `Module01/Module01_Slides.html`

- **Module 02 – Discrete-Time Methods**  
  `Module02/Module02_Slides.html`

- **Module 03 – Continuous-Time Methods**  
  `Module03/Module03_Slides.html`

- **Module 04 – Fundamentals of Machine Learning**  
  `Module04/Module04_Slides.html`

- **Module 05 – The Deep Policy Iteration Method**  
  `Module05/Module05_Slides.html`

---

## 💻 Interactive Pluto Notebooks

Static HTML previews of the notebooks live in the `notebooks/` folder:

- `notebooks/NB_ThreeChallenges.html`
- `notebooks/NB_BlackScholes.html`
- `notebooks/NB_FittingDNN.html`
- `notebooks/NB_TwoTrees.html`

The original Pluto notebooks are in `src/Module0X/` and can be run locally.

To launch Pluto and open notebooks:

```julia
using Pluto
Pluto.run()
````

Or run a specific notebook directly, for example the Module 2 notebook:
```
pluto run src/Module02/NB_ThreeChallenges.jl
```

## 🔍 Replication Code

All replication code for figures, examples, and numerical methods discussed in the course is located under `src/`.

The codebase covers, among other things:

- Value Function Iteration (VFI)  
- Policy Function Iteration (PFI)  
- Endogenous Gridpoint Method (EGM)  
- Deep Policy Iteration (DPI)  
- Neural-network approximations to value and policy functions  
- Automatic differentiation, stochastic calculus tools, and related utilities  

The structure is modular so that users can reuse individual components in their own projects.

---

## 📄 License and Use

The material in this repository is intended for **educational and research purposes**.

