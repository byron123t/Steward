
# Steward
A minimalized version of the codebase used in the paper ([https://arxiv.org/pdf/2409.15441](https://arxiv.org/pdf/2409.15441)). requirements.txt contains the package dependencies.

### Installation
Using python version 3.9.18 and conda
```
conda create -n steward python=3.9
conda activate steward
pip install -r requirements.txt
```
Run `playwright install`

### Running
Edit `run.sh` to your desires
Insert your OpenAI API key or Azure OpenAI key + URL in `sensitive.py`. If using Azure, you need to modify the `self.deployment` strings to match your Azure model deployment names (lines 15 & 17) in `API.py`. You also need to modify lines 29-31 in `smart_runtime.py`, adding `, mode='azure'` to any models using Azure.
Run `./run.sh` from within the repository


### Citation

@article{tang2024steward,
  title={Steward: Natural language web automation},
  author={Tang, Brian and Shin, Kang G},
  journal={arXiv preprint arXiv:2409.15441},
  year={2024}
}
