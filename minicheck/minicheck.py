# Adapt code from https://github.com/yuh-zha/AlignScore/tree/main

import sys
sys.path.append("..")

from minicheck.inference import Inferencer, LLMCheck
from typing import List
import numpy as np


class MiniCheck:
    def __init__(self, model_name='Bespoke-MiniCheck-7B', max_input_length=None, batch_size=16, cache_dir=None, tensor_parallel_size=1, max_tokens=1) -> None:

        '''
        Model Options: 
        We have 5 models available for MiniCheck.

        Note:
        (1) MiniCheck-Flan-T5-Large (770M) is the best fack-checking model with size < 1B and reaches GPT-4 performance.
        (2) Bespoke-MiniCheck-7B is the most performant fact-checking model in the MiniCheck series AND
            it outperforms ALL exisiting specialized fact-checkers and off-the-shelf LLMs regardless of size.

        Throughput:
        We automatically speedup Bespoke-MiniCheck-7B inference with vLLM. Based on our test on a single A6000 (48 VRAM), 
        both Bespoke-MiniCheck-7B with vLLM and MiniCheck-Flan-T5-Large have throughputs > 500 docs/min.
        '''

        assert model_name in ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B'], \
            "model_name must be one of ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B']"

        
        if model_name in ['roberta-large', 'deberta-v3-large', 'flan-t5-large']:
            self.model = Inferencer(
                model_name=model_name, 
                batch_size=batch_size, 
                max_input_length=max_input_length,
                cache_dir=cache_dir
            )
        elif model_name == 'Bespoke-MiniCheck-7B':
            self.model = LLMCheck(
                model_id=model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
            )
        

    def score(self, docs: List[str], claims: List[str], chunk_size=None) -> List[float]:
        '''
        pred_labels: 0 / 1 (0: unsupported, 1: supported)
        max_support_probs: the probability of "supported" for the chunk that determin the final pred_label
        used_chunks: divided chunks of the input document
        support_prob_per_chunk: the probability of "supported" for each chunk
        '''

        assert isinstance(docs, list) or isinstance(docs, np.ndarray), "docs must be a list or np.ndarray"
        assert isinstance(claims, list) or isinstance(claims, np.ndarray), "claims must be a list or np.ndarray"  

        if isinstance(self.model, Inferencer):
            return self._score_inferencer(docs, claims, chunk_size)
        elif isinstance(self.model, LLMCheck):
            return self._score_llmcheck(docs, claims, chunk_size)

    
    def _score_inferencer(self, docs, claims, chunk_size):

        if chunk_size and isinstance(chunk_size, int) and chunk_size > 0:
            self.model.chunk_size = chunk_size
        else:
            self.model.chunk_size = 500 if self.model.model_name == 'flan-t5-large' else 400

        max_support_prob, used_chunk, support_prob_per_chunk = self.model.fact_check(docs, claims)
        pred_label = [1 if prob > 0.5 else 0 for prob in max_support_prob]

        return pred_label, max_support_prob, used_chunk, support_prob_per_chunk
    
    def _score_llmcheck(self, docs, claims, chunk_size):
        return self.model.score(docs, claims, chunk_size)
    

if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    doc = "A group of students gather in the school library to study for their upcoming final exams."
    claim_1 = "The students are preparing for an examination."
    claim_2 = "The students are on vacation."

    # model_name can be one of:
    # ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B']

    # bespokelabs/Bespoke-MiniCheck-7B will be auto-downloaded from Huggingface for the first time
    # Bespoke-MiniCheck-7B is the most performant fact-checking model in the MiniCheck series AND
    # it outperforms ALL exisiting specialized fact-checkers and off-the-shelf LLMs regardless of size.
    scorer = MiniCheck(model_name='Bespoke-MiniCheck-7B', cache_dir='./ckpts')
    pred_label, raw_prob, _, _ = scorer.score(docs=[doc, doc], claims=[claim_1, claim_2])

    print(pred_label) # [1, 0]
    print(raw_prob)   # [0.9840446675150499, 0.010986349594852094]