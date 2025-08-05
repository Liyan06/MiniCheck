# Adapt code from https://github.com/yuh-zha/AlignScore/tree/main

from minicheck.inference import Inferencer, LLMCheck
from typing import List
import numpy as np


class MiniCheck:
    def __init__(self, model_name='Bespoke-MiniCheck-7B', max_model_len=None, batch_size=16, cache_dir=None, tensor_parallel_size=1, max_tokens=1, enable_prefix_caching=False) -> None:

        '''
        Parameters:
        -----------
        model_name : str, optional (default='Bespoke-MiniCheck-7B')
            The name of the model to use. Options are:
            - 'roberta-large'
            - 'deberta-v3-large'
            - 'flan-t5-large'
            - 'Bespoke-MiniCheck-7B'
            - 'Granite-Guardian-3.3-8B'
            Note: 'Bespoke-MiniCheck-7B' is the most performant fact-checking model in the MiniCheck series.
        
        max_model_len : int or None, optional (default=None)
            The maximum input length for the model. If None, we use the following default values. 
                - 'roberta-large'
                    Default: 512
                - 'deberta-v3-large'
                    Default: 2048
                - 'flan-t5-large'
                    Default: 2048
                - 'Bespoke-MiniCheck-7B'
                    Default: 32768
                - 'Granite-Guardian-3.3-8B'
                    Default: 32768
            For 'Bespoke-MiniCheck-7B', if you have a GPU with low VRAM and get the following:
                "ValueError: The model's max seq len (XXXX) is larger than the maximum number of 
                tokens that can be stored in KV cache (YYYY). Try increasing `gpu_memory_utilization` 
                or decreasing `max_model_len` when initializing the engine."
            Please consider setting `max_model_len` to a smaller value, recommendation would be a value
            slightly less than based on observations.

        batch_size : int, optional (default=16)
            The batch size for inference. Only applicable for non-LLM-based MiniCheck models.
            'Bespoke-MiniCheck-7B' automatically use dynamic batching.

        cache_dir : str or None, optional (default=None)
            The directory to cache the model. If None, the default cache directory is used.

        tensor_parallel_size : int, optional (default=1)
            The number of GPUs to use for inference. Only applicable for 'Bespoke-MiniCheck-7B'.

        max_tokens : int, optional (default=1)
            The maximum number of tokens to generate. Only applicable for 'Bespoke-MiniCheck-7B'.
            For 'Granite-Guardian-3.3-8B' used max_tokens=2048

        enable_prefix_caching : bool, optional (default=False)
            Whether to enable prefix caching for 'Bespoke-MiniCheck-7B'. This can improve performance
            when using the same document chunk to fact-check different claims.

        Note:
        (1) MiniCheck-Flan-T5-Large (770M) is the best fack-checking model with size < 1B and reaches GPT-4 performance.
        (2) Bespoke-MiniCheck-7B is the most performant fact-checking model in the MiniCheck series AND
            it outperforms ALL exisiting specialized fact-checkers and off-the-shelf LLMs regardless of size.

        Throughput:
        We automatically speedup Bespoke-MiniCheck-7B inference with vLLM. Based on our test on a single A6000 (48 VRAM), 
        both Bespoke-MiniCheck-7B with vLLM and MiniCheck-Flan-T5-Large have throughputs > 500 docs/min.

        Automatic Prefix Caching for Bespoke-MiniCheck-7B:
        If you use the same document to fact-check different claims, APC allows vLLM to process the document only once, 
        and all future claims can avoid recomputing this document by reusing its KV cache. This allows vLLM to serve 
        future grounded fact-checking with much higher throughput and much lower latency.
        '''

        assert model_name in ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B', 'Granite-Guardian-3.3-8B'], \
            "model_name must be one of ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B', 'Granite-Guardian-3.3-8B']"

        
        if model_name in ['roberta-large', 'deberta-v3-large', 'flan-t5-large']:
            self.model = Inferencer(
                model_name=model_name, 
                batch_size=batch_size, 
                max_model_len=max_model_len,
                cache_dir=cache_dir
            )
        elif model_name == 'Bespoke-MiniCheck-7B':
            self.model = LLMCheck(
                model_id=model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len
            )
        elif model_name == 'Granite-Guardian-3.3-8B':
            if not max_tokens or max_tokens<2048:
                print("For Granite Guardian 3.3 - fixing the max_tokens to be 2048")
                max_tokens=2048

            self.model = LLMCheck(
                model_id=model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len
            )
        

    def score(self, docs: List[str], claims: List[str], chunk_size=None) -> List[float]:
        '''
        Parameters:
        -----------
        chunk_size : int or None, optional (default=None)
            The size of the chunk for long documents. The document will be splitted in to
            chunks of size chunk_size. If None, the default chunk size is used.
            - 'roberta-large'
                Default: 400
            - 'deberta-v3-large'
                Default: 400
            - 'flan-t5-large'
                Default: 500
            - 'Bespoke-MiniCheck-7B'
                Default: 32768-300

        Returns:
        -----------
        pred_labels : 0 / 1 (0: unsupported, 1: supported)
        max_support_probs : the probability of "supported" for the chunk that determin the final pred_label
        used_chunks : divided chunks of the input document
        support_prob_per_chunk : the probability of "supported" for each chunk
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
