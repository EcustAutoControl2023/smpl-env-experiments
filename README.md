## Documentation

### Installation

```shell
git clone --recursive git@github.com:EcustAutoControl2023/smpl-env-experiments.git && cd smpl-env-experiments &&  git submodule update --recursive --remote

```

Then install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
uv sync
```

### Usage
Activate venv
```shell
source .venv/bin/activate
```

Demo test
```shell
cd test && uv run ./PenSimEnvTest.py
```
