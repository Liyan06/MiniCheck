# MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents

Authors: Liyan Tang, Philippe Laban, Greg Durrett

Please check out our work [here](https://arxiv.org/pdf/2404.10774.pdf) 📃

<p align="center">
    <img src="./images/main-figure.png" width="360">
</p>


## *LLM-AggreFact* Benchmark

### Description

LLM-AggreFact is a fact verification benchmark. It aggregates 10 of the most up-to-date publicly available datasets on factual consistency evaluation across both closed-book and grounded generation settings. In LLM-AggreFact:
1. Documents come from diverse sources, including Wikipedia paragraphs, interviews, web text, covering domains such as news, dialogue, science, and healthcare.
2. Claims to be verified are mostly generated from recent generative models (except for one dataset of human-written claims), *without any human intervention in any format, such as injecting certain error types into model-generated claims*.

### Benchmark Access

Our Benchmark is available on HuggingFace 🤗 More benchmark details can be found [here](https://huggingface.co/datasets/lytang/LLM-AggreFact).

```python
from datasets import load_dataset
dataset = load_dataset("lytang/LLM-AggreFact")
```

The benchmark contains the following fields:

|Field| Description |
|--|--|
|dataset| One of the 10 datasets in the benchmark|
|doc| Document used to check the corresponding claim|
|claim| Claim to be checked by the corresponding document|
|label| 1 if the claim is supported, 0 otherwise|

## *MiniCheck* Model Evaluation Demo

Please first clone our GitHub Repo and install necessary packages from `requirements.txt`. 

Our MiniCheck models are available on HuggingFace 🤗 More model details can be found from this [collection](https://huggingface.co/collections/lytang/minicheck-and-llm-aggrefact-661c5d387082ad0b433dec65). Below is a simple use case of MiniCheck. MiniCheck models will be automatically downloaded from Huggingface for the first time and cached in the specified directory.


```python
from minicheck.minicheck import MiniCheck

doc = "A group of students gather in the school library to study for their upcoming final exams."
claim_1 = "The students are preparing for an examination."
claim_2 = "The students are on vacation."

# model_name can be one of ['roberta-large', 'deberta-v3-large', 'flan-t5-large']
# lytang/MiniCheck-Flan-T5-Large will be auto-downloaded from Huggingface for the first time
scorer = MiniCheck(model_name='flan-t5-large', device=f'cuda:0', cache_dir='./ckpts')
pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])

print(pred_label) # [1, 0]
print(raw_prob)   # [0.9805923700332642, 0.007121307775378227]
```

A detailed walkthrough of the evaluation process on LLM-Aggrefact and replication of the results is available in this notebook: [inference-example-demo.ipynb](./inference-example-demo.ipynb).


## Synthetic Data Generation 

Available soon.