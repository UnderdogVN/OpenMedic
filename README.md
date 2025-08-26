# OpenMedic
OpenMedic is an open-source Python framework designed to accelerate the development of deep learning models in the medical domain. Built for researchers, engineers, and healthcare innovators, OpenMedic provides modular components, reusable pipelines, and streamlined tools for building, training, and deploying models for tasks such as medical image segmentation, classification, and diagnosis.

## 1. Installation
### 1.1. Install with `uv`
`uv` is a super fast Python package manager (alternative to pip and poetry), built in Rust. It can handle installing, resolving, and locking dependencies.

Please follow the [instruction](https://docs.astral.sh/uv/getting-started/installation/) to install `uv`.

Setup env:
- Set `__pycache__` location if you do not want `__pycache__` directories to appear in the repo.
```bash
export PYTHONPYCACHEPREFIX=~/.cache/Python # for example
```

- Initalizes python env with dependencies. Now OpenMedic just supports `python3.10.18`.
```bash
uv venv --python 3.10.18
```

- Activates python env.
```bash
source .venv/bin/activate
```
  - If you are using a shell different than `sh`/`bash`/`zsh`, you may need to source the correct file. For example, if you are using Fish:
    ```fish
    source .venv/bin/activate.fish
    ```

- Installs `openmedic`:
```bash
uv pip install -e .
```

- Test env:
```bash
openmedic --version
```

*Expected result:* openmedic, version <>

## 2. Development
### 2.0 (Optional) Run GitHub Actions locally using either option below:
  - <ins>Option A:</ins> Use Nix:
      - Install [Nix](https://nixos.org/download/)
      - Enable Nix Flakes by adding to `experimental-features = nix-command flakes` to `~/.config/nix/nix.conf`:
      - Run `nix develop`
  - <ins>Option B:</ins> Install [act](https://nektosact.com/) any way you like 
