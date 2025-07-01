# OpenMedic
OpenMedic is an open-source Python framework designed to accelerate the development of deep learning models in the medical domain. Built for researchers, engineers, and healthcare innovators, OpenMedic provides modular components, reusable pipelines, and streamlined tools for building, training, and deploying models for tasks such as medical image segmentation, classification, and diagnosis.

## 1. Installation
### 1.1. Install with `uv`
`uv` is a super fast Python package manager (alternative to pip and poetry), built in Rust. It can handle installing, resolving, and locking dependencies.

Please follow the [instruction](https://docs.astral.sh/uv/getting-started/installation/) to install `uv`.

Setup env:

Initalizes python env with dependencies. Now OpenMedic just supports `python3.10`.
```bash
uv sync
```

Installs `openmedic`:
```bash
uv pip install -e .
```

Test env:
```bash
openmedic --version
```

*Expected result:* openmedic, version <>