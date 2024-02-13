
# Steward
A minimalized version of the codebase used in the paper. requirements.txt contains the package dependencies.

### Installation
Using python version 3.9.18 and conda
```
conda create -n MY_ENV_NAME python=3.9
conda activate MY_ENV_NAME
pip install -r requirements.txt
```
Run `playwright install`

### Running
Edit `run.sh` to your desires
Insert your OpenAI API key or Azure OpenAI key + URL in `sensitive.py`. If using Azure, you need to modify the `self.deployment` strings to match your Azure model deployment names (lines 15 & 17) in `API.py`. You also need to modify lines 29-31 in `smart_runtime.py`, adding `, mode='azure'` to any models using Azure.
Run `./run.sh` from within the repository
