## Documentation

### Installation
necessary system packages(if you are using Ubuntu2004; otherwise you should install the right version of each library)
```shell
sudo apt install gcc g++ cmake libboost-all-dev libomp-dev -y
```
The recommend version of each library is list below:
- boost-all-dev: 1.71.0
- openmp: 10.0.0
- gcc9
- g++9

clone this repository
```shell
git clone --recursive git@github.com:EcustAutoControl2023/smpl-env-experiments.git && cd smpl-env-experiments &&  git submodule update --recursive --remote

```

Then install [uv](https://docs.astral.sh/uv/getting-started/installation/), which takes care of all dependences, just run one command below.

```shell
uv sync
```
Virtual environment is under `.venv` folder.

### Usage
Activate venv
```shell
source .venv/bin/activate
```

Demo test
```shell
cd test && uv run ./PenSimEnvTest.py
```
