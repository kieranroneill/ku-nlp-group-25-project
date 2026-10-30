<h1 align="center">
  NLP Group 25 Project
</h1>

---

### Table of contents

* [1. Overview](#-1-overview)
  - [1.1. Project structure](#11-project-structure)
* [2. Usage](#-2-usage)
  - [2.1. Requirements](#21-requirements)
  - [2.2. Installation](#22-installation)
  - [2.3. Validate Challenge Set](#23-validate-challenge-set)
* [3. Appendix](#-3-appendix)
  - [3.1. Useful commands](#31-useful-commands)
  - [3.2. Useful links](#32-useful-links)

## 🔭 1. Overview

### 1.1. Project structure

```text
.
├─ data/                            <-- Data files
│   └── ...
├─ docs/                            <-- LaTex files for report
│   ├── main.tex
│   └── ...
├─ genai-logs/                      <-- Generative AI logs for each student
│   ├── <ku-username>/
│   │   └── ...
│   └── ...
├─ notebooks/                       <-- Notebooks to run analysis
│   ├── notebook.ipynb
│   └── ...
├─ outputs/                         <-- Outputs of models for analysis
│   └── ...
├─ scripts/                         <-- Stand-alone scripts
│   └── ...
├─ src/                             <-- Internal Python code and packages
│   ├── <package-name>/
│   │   ├── __init__.py
│   │   └── ...
│   ├── __init__.py
│   └── ...
├─ tests/                           <-- Unit tests
│   ├── <package-name>/
│   │   ├── test_*.py
│   │   └── ...
│   └── ...
├── .editorconfig
├── .gitignore
├── .python-version
├── pyproject.toml
├── pytest.ini
├── README.md
├── uv.lock
└── ...
```

<sup>[Back to top ^][table-of-contents]</sup>

## 🪄 2. Usage

#### 2.1. Requirements

- [Python v3.12+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

<sup>[Back to top ^][table-of-contents]</sup>

#### 2.2. Installation

Install the dependencies and set up the Python virtual environment using:

```bash
$ uv sync
```

<sup>[Back to top ^][table-of-contents]</sup>

### 2.3. Validate Challenge Set

Included in `scripts/` is a stand-alone Python script to validate a challenge set.

Once you have set-up the environment in the [above](#22-installation) step, you can validate the challenge set using:

```bash
$ uv run scripts/validate_challenge.py <path-to-challenge-set>
```

<sup>[Back to top ^][table-of-contents]</sup>

## 📑 3. Appendix

### 3.1. Useful commands

| Command         | Description                                         |
|-----------------|-----------------------------------------------------|
| `uv run pytest` | Run unit tests. See `pytest.ini` for configuration. |
| `uv sync`       | Installs the dependencies.                          |

<sup>[Back to top ^][table-of-contents]</sup>

### 3.2. Useful links

| Link                                                                                                                     | Description                                                             |
|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [Overleaf](https://www.overleaf.com/project/6ab107696f1f5e9e7040155b)                                                    | The Overleaf project that to collaborate.                               |

<sup>[Back to top ^][table-of-contents]</sup>

<!-- links -->

[table-of-contents]: #table-of-contents
