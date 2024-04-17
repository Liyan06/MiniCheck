# Adapt code from https://github.com/yuh-zha/AlignScore/tree/main

import sys
sys.path.append("..")

from minicheck.inference import Inferencer
from typing import List
import numpy as np


class MiniCheck:
    def __init__(self, model_name='flan-t5', device='cuda:0', chunk_size=None, max_input_length=None, batch_size=16, cache_dir=None) -> None:

        assert model_name in ['roberta-large', 'deberta-v3-large', 'flan-t5-large'], \
            "model_name must be one of ['roberta-large', 'deberta-v3-large', 'flan-t5-large']"

        self.model = Inferencer(
            model_name=model_name, 
            batch_size=batch_size, 
            device=device,
            chunk_size=chunk_size,
            max_input_length=max_input_length,
            cache_dir=cache_dir
        )

    def score(self, docs: List[str], claims: List[str]) -> List[float]:
        '''
        pred_labels: 0 / 1 (0: unsupported, 1: supported)
        max_support_probs: the probability of "supported" for the chunk that determin the final pred_label
        used_chunks: divided chunks of the input document
        support_prob_per_chunk: the probability of "supported" for each chunk
        '''

        assert isinstance(docs, list) or isinstance(docs, np.ndarray), "docs must be a list or np.ndarray"
        assert isinstance(claims, list) or isinstance(claims, np.ndarray), "claims must be a list or np.ndarray"  

        max_support_prob, used_chunk, support_prob_per_chunk = self.model.fact_check(docs, claims)
        pred_label = [1 if prob > 0.5 else 0 for prob in max_support_prob]

        return pred_label, max_support_prob, used_chunk, support_prob_per_chunk
    

if __name__ == '__main__':

    model_name = 'flan-t5-large'   # ['roberta-large', 'deberta-v3-large', 'flan-t5-large']
    scorer = MiniCheck(model_name=model_name, device=f'cuda:0', cache_dir='./ckpts')

    doc = "A group of students gather in the school library to study for their upcoming final exams."
    claim_1 = "The students are preparing for an examination."
    claim_2 = "The students are on vacation."

    # model_name can be one of ['roberta-large', 'deberta-v3-large', 'flan-t5-large']
    # lytang/MiniCheck-Flan-T5-Large will be auto-downloaded from Huggingface for the first time
    scorer = MiniCheck(model_name='flan-t5-large', device=f'cuda:0', cache_dir='./ckpts')
    pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])

    print(pred_label) # [1, 0]
    print(raw_prob)   # [0.9805923700332642, 0.007121307775378227]