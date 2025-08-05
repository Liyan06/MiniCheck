# Adapt code from https://github.com/yuh-zha/AlignScore/tree/main

from nltk.tokenize import sent_tokenize
import numpy as np
import torch
import nltk
import random
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import os
from minicheck.utils import SYSTEM_PROMPT, USER_PROMPT
from typing import List


def sent_tokenize_with_newlines(text):
    blocks = text.split('\n')
    
    tokenized_blocks = [sent_tokenize(block) for block in blocks]
    tokenized_text = []
    for block in tokenized_blocks:
        tokenized_text.extend(block)
        tokenized_text.append('\n')  

    return tokenized_text[:-1]  


class Inferencer():
    def __init__(self, model_name, max_model_len, batch_size, cache_dir) -> None:
        
        self.model_name = model_name

        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

        if model_name == 'flan-t5-large':
            ckpt = 'lytang/MiniCheck-Flan-T5-Large'
            self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, cache_dir=cache_dir, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt, cache_dir=cache_dir)

            self.max_model_len=2048 if max_model_len is None else max_model_len
            self.max_output_length = 256
        
        else:
            if model_name == 'roberta-large':
                ckpt = 'lytang/MiniCheck-RoBERTa-Large'
                self.max_model_len=512 if max_model_len is None else max_model_len

            elif model_name == 'deberta-v3-large':
                ckpt = 'lytang/MiniCheck-DeBERTa-v3-Large'
                self.max_model_len=2048 if max_model_len is None else max_model_len
                
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            config = AutoConfig.from_pretrained(ckpt, num_labels=2, finetuning_task="text-classification", revision='main', token=None, cache_dir=cache_dir)
            config.problem_type = "single_label_classification"

            self.tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True, revision='main', token=None, cache_dir=cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                ckpt, config=config, revision='main', token=None, ignore_mismatched_sizes=False, cache_dir=cache_dir, device_map="auto")
        
        self.model.eval()
        self.batch_size = batch_size
        self.softmax = nn.Softmax(dim=-1)

    def inference_example_batch(self, doc: list, claim: list):
        """
        inference a example,
        doc: list
        claim: list
        using self.inference to batch the process
        """

        assert len(doc) == len(claim), "doc must has the same length with claim!"

        max_support_probs = []
        used_chunks = []
        support_prob_per_chunk = []
        
        for one_doc, one_claim in tqdm(zip(doc, claim), desc="Evaluating", total=len(doc)):
            output = self.inference_per_example(one_doc, one_claim)
            max_support_probs.append(output['max_support_prob'])
            used_chunks.append(output['used_chunks'])
            support_prob_per_chunk.append(output['support_prob_per_chunk'])
        
        return {
            'max_support_probs': max_support_probs,
            'used_chunks': used_chunks,
            'support_prob_per_chunk': support_prob_per_chunk
        }

    def inference_per_example(self, doc:str, claim: str):
        """
        inference a example,
        doc: string
        claim: string
        using self.inference to batch the process
        """
        def chunks(lst, n):
            """Yield successive chunks from lst with each having approximately n tokens.

            For flan-t5, we split using the white space;
            For roberta and deberta, we split using the tokenization.
            """
            if self.model_name == 'flan-t5-large':
                current_chunk = []
                current_word_count = 0
                for sentence in lst:
                    sentence_word_count = len(sentence.split())
                    if current_word_count + sentence_word_count > n:
                        yield ' '.join(current_chunk)
                        current_chunk = [sentence]
                        current_word_count = sentence_word_count
                    else:
                        current_chunk.append(sentence)
                        current_word_count += sentence_word_count
                if current_chunk:
                    yield ' '.join(current_chunk)
            else:
                current_chunk = []
                current_token_count = 0
                for sentence in lst:
                    sentence_word_count = len(self.tokenizer(
                        sentence, padding=False, add_special_tokens=False, 
                        max_length=self.max_model_len, truncation=True)['input_ids'])
                    if current_token_count + sentence_word_count > n:
                        yield ' '.join(current_chunk)
                        current_chunk = [sentence]
                        current_token_count = sentence_word_count
                    else:
                        current_chunk.append(sentence)
                        current_token_count += sentence_word_count
                if current_chunk:
                    yield ' '.join(current_chunk)

        doc_sents = sent_tokenize_with_newlines(doc)
        doc_sents = doc_sents or ['']

        doc_chunks = [chunk.replace(" \n ", '\n').strip() for chunk in chunks(doc_sents, self.chunk_size)]
        doc_chunks = [chunk for chunk in doc_chunks if chunk != '']

        '''
        [chunk_1, chunk_2, chunk_3, chunk_4, ...]
        [claim]
        '''
        claim_repeat = [claim] * len(doc_chunks)
        
        output = self.inference(doc_chunks, claim_repeat)
        
        return output

    def inference(self, doc, claim):
        """
        inference a list of doc and claim

        Standard aggregation (max) over chunks of doc

        Note: We do not have any post-processing steps for 'claim'
        and directly check 'doc' against 'claim'. If there are multiple 
        sentences in 'claim'. Sentences are not splitted and are checked 
        as a single piece of text.
        
        If there are multiple sentences in 'claim', we suggest users to 
        split 'claim' into sentences beforehand and prepares data like 
        (doc, claim_1), (doc, claim_2), ... for a multi-sentence 'claim'.

        **We leave the user to decide how to aggregate the results from multiple sentences.**

        Note: AggreFact-CNN is the only dataset that contains three-sentence 
        summaries and have annotations on the whole summaries, so we do not 
        split the sentences in each 'claim' during prediciotn for simplicity. 
        Therefore, for this dataset, our result is based on treating the whole 
        summary as a single piece of text (one 'claim').

        In general, sentence-level prediciton performance is better than that on 
        the full-response-level.
        """

        if isinstance(doc, str) and isinstance(claim, str):
            doc = [doc]
            claim = [claim]
        
        batch_input, _, batch_org_chunks = self.batch_tokenize(doc, claim)

        label_probs_list = []
        used_chunks = []

        for mini_batch_input, batch_org_chunk in zip(batch_input, batch_org_chunks):

            mini_batch_input = {k: v.to(self.model.device) for k, v in mini_batch_input.items()}

            with torch.no_grad():

                if self.model_name == 'flan-t5-large':
                    
                    decoder_input_ids = torch.zeros((mini_batch_input['input_ids'].size(0), 1), dtype=torch.long).to(self.model.device)
                    outputs = self.model(input_ids=mini_batch_input['input_ids'], attention_mask=mini_batch_input['attention_mask'], decoder_input_ids=decoder_input_ids)
                    logits = outputs.logits.squeeze(1)

                    # 3 for no support and 209 for support
                    label_logits = logits[:, torch.tensor([3, 209])].cpu()
                    label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
                
                else:

                    outputs = self.model(**mini_batch_input)
                    logits = outputs.logits
                    label_probs = F.softmax(logits, dim=1)    

                label_probs_list.append(label_probs)
                used_chunks.extend(batch_org_chunk)

        label_probs = torch.cat(label_probs_list)
        support_prob_per_chunk = label_probs[:, 1].cpu().numpy()
        max_support_prob = label_probs[:, 1].max().item()
        
        return {
            'max_support_prob': max_support_prob,
            'used_chunks': used_chunks,
            'support_prob_per_chunk': support_prob_per_chunk
        }

    def batch_tokenize(self, doc, claim):
        """
        input doc and claims are lists
        """
        assert isinstance(doc, list) and isinstance(claim, list)
        assert len(doc) == len(claim), "doc and claim should be in the same length."

        original_text = [self.tokenizer.eos_token.join([one_doc, one_claim]) for one_doc, one_claim in zip(doc, claim)]

        batch_input = []
        batch_concat_text = []
        batch_org_chunks = []
        for mini_batch in self.chunks(original_text, self.batch_size):
            if self.model_name == 'flan-t5-large':
                model_inputs = self.tokenizer(
                    ['predict: ' + text for text in mini_batch], 
                    max_length=self.max_model_len, 
                    truncation=True, 
                    padding=True, 
                    return_tensors="pt"
                ) 
            else:
                model_inputs = self.tokenizer(
                    [text for text in mini_batch], 
                    max_length=self.max_model_len, 
                    truncation=True, 
                    padding=True, 
                    return_tensors="pt"
                ) 
            batch_input.append(model_inputs) 
            batch_concat_text.append(mini_batch)  
            batch_org_chunks.append([item[:item.find('</s>')] for item in mini_batch]) 

        return batch_input, batch_concat_text, batch_org_chunks

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def fact_check(self, doc, claim):

        outputs = self.inference_example_batch(doc, claim)
        return outputs['max_support_probs'], outputs['used_chunks'], outputs['support_prob_per_chunk']


class LLMCheck:

    def __init__(self, model_id, tensor_parallel_size=1, max_tokens=1, cache_dir=None, enable_prefix_caching=False, max_model_len=None):
        from vllm import LLM, SamplingParams

        import logging
        logging.basicConfig(
            level=logging.INFO,  
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

        logging.info("Reminder: Please set the CUDA device before initializing the LLMCheck object.")

        if model_id == 'Bespoke-MiniCheck-7B':
            self.model_id = 'bespokelabs/Bespoke-MiniCheck-7B'
            self.operating_mode="bespoke"
        elif model_id == 'Granite-Guardian-3.3-8B':
            self.model_id = 'ibm-granite/granite-guardian-3.3-8b'
            self.operating_mode="gg_hybrid"
        else:
            raise ValueError("model_id must be 'Bespoke-MiniCheck-7B'")

        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.max_model_len = 32768 if max_model_len is None else max_model_len # max input length (prompt + doc)
        self.default_chunk_size = self.max_model_len - 300 # reserve some space (hard coded) for the claim to be checked
        self.cache_dir = cache_dir

        self.user_prompt = USER_PROMPT
        self.system_prompt = SYSTEM_PROMPT
        self.enable_prefix_caching = enable_prefix_caching

        # Check if CUDA is available and get compute capability
        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] >= 8:
                self.dtype = torch.bfloat16
                logging.info("Using bfloat16 for LLM initialization.")
            else:
                self.dtype = torch.float16
                logging.info(f"GPU compute capability {compute_capability} < 8.0. Using float16 for LLM initialization.")
        else:
            if torch.cpu.is_available() and hasattr(torch.cpu, 'is_bf16_supported') and torch.cpu.is_bf16_supported():
                self.dtype = torch.bfloat16
                logging.info("CUDA not available. Using bfloat16 on CPU.")
            else:
                self.dtype = torch.float32
                logging.info("CUDA not available and CPU doesn't support bfloat16. Using float32 for LLM initialization.")
        
        self.llm = LLM(
            model=self.model_id, 
            dtype=self.dtype, 
            download_dir=self.cache_dir,
            trust_remote_code=True if self.model_id == 'bespokelabs/Bespoke-MiniCheck-7B' else False,
            tensor_parallel_size=self.tensor_parallel_size,
            seed=2024,
            max_model_len=self.max_model_len,   # need to be adjusted based on the GPU memory available
            enable_prefix_caching=self.enable_prefix_caching
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.padding_side = "left"
        terminators = [
            self.tokenizer.eos_token_id,
        ]
        converted_token = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if converted_token is not None:
            terminators.append(converted_token)

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_tokens,
            stop_token_ids=terminators,
            logprobs=5
        )


    def sent_tokenize_with_newlines(self, text):
        blocks = text.split('\n')
        
        tokenized_blocks = [sent_tokenize(block) for block in blocks]
        tokenized_text = []
        for block in tokenized_blocks:
            tokenized_text.extend(block)
            tokenized_text.append('\n')  

        return tokenized_text[:-1] 
    

    def apply_chat_template(self, doc, claim):
        if self.operating_mode=="bespoke":
            user_prompt = self.user_prompt.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim)
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        elif self.operating_mode=="gg_hybrid":
            documents = [{'doc_id':'0', 'text': doc}]
            messages = [{"role": "assistant", "content": claim}]
            guardian_config = {"criteria_id": "groundedness"}
            text = self.tokenizer.apply_chat_template(messages, guardian_config = guardian_config, documents=documents, think=True, tokenize=False, add_generation_prompt=True)
        return text

    
    def get_support_prob(self, response):
        """probs from vllm inference"""
        import math
        support_prob = 0

        for token_prob in response.outputs[0].logprobs[0].values():
            decoded_token = token_prob.decoded_token
            if decoded_token.lower() == 'yes': 
                support_prob += math.exp(token_prob.logprob)
        
        return support_prob
    
    def get_support_prob_hybrid_gg(self, response, marker="score"):
        """probs from vllm inference"""
        response_text = response.outputs[0].text.lower()
        try:
            support_prob=1.0 if f"<{marker}> no </{marker}>" in response_text else 0.0
        except Exception as e:
            print("Error:", e)
            support_prob = random.random()
        return support_prob


    def get_all_chunks_per_doc(self, doc, claim):
    
        def chunks(lst, n):
            """Yield successive chunks from lst with each having approximately n tokens.
            """
            current_chunk = []
            current_word_count = 0
            for sentence in lst:
                sentence_word_count = len(self.tokenizer(sentence, add_special_tokens=False)['input_ids'])
                if current_word_count + sentence_word_count > n:
                    yield ' '.join(current_chunk)
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count
                else:
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count
            if current_chunk:
                yield ' '.join(current_chunk)

        if doc in self.doc_chunk_cache:
            doc_chunks = self.doc_chunk_cache[doc]
        else:
            doc_sents = self.sent_tokenize_with_newlines(doc)
            doc_sents = doc_sents or ['']
    
            doc_chunks = [chunk.replace(" \n ", '\n').strip() for chunk in chunks(doc_sents, self.chunk_size)]
            doc_chunks = [chunk for chunk in doc_chunks if chunk != '']
            self.doc_chunk_cache[doc] = doc_chunks
        if len(doc_chunks) == 0:
            doc_chunks = [''] 

        claim_repeat = [claim] * len(doc_chunks)

        return {'doc_chunks': doc_chunks, 'claim_repeat': claim_repeat}


    def score(self, docs: List[str], claims: List[str], chunk_size=None) -> List[float]:

        self.doc_chunk_cache = {}
        self.chunk_size = chunk_size if chunk_size else self.default_chunk_size

        assert self.chunk_size < self.max_model_len, \
            "chunk_size must be less than max_model_len so that MiniCheck can process the claim"

        all_prompts = []
        doc_claim_indices = []
        
        for index, (doc, claim) in tqdm(enumerate(zip(docs, claims)), total=len(docs), desc="Tokenizing"):
            chunks = self.get_all_chunks_per_doc(doc, claim)
            doc_chunks = chunks['doc_chunks']
            claim_repeat = chunks['claim_repeat']

            # Split the claim into individual sentences
            claim_sentences = self.split_into_sentences(claim)

            # Apply SentenceFusion for granular introspection
            prompts = []
            for doc_chunk in doc_chunks:
                for sentence in claim_sentences:
                    prompt = self.apply_chat_template(doc_chunk, sentence)
                    prompts.append(prompt)
            all_prompts.extend(prompts)
            doc_claim_indices.extend([index] * len(prompts))

        responses = self.llm.generate(all_prompts, self.sampling_params) 
        if self.operating_mode=="bespoke":
            probs_per_chunk_sentence = [self.get_support_prob(responses[idx]) for idx in range(len(responses))]
        elif self.operating_mode=="gg_hybrid":
            probs_per_chunk_sentence = [self.get_support_prob_hybrid_gg(responses[idx]) for idx in range(len(responses))]

        result_dict = {}
        for index, prob_per_chunk_sentence in zip(doc_claim_indices, probs_per_chunk_sentence):
            if index not in result_dict:
                result_dict[index] = []
            result_dict[index].append(prob_per_chunk_sentence)

        probs_per_doc_claim_pair = [result_dict[index] for index in range(len(docs))] 
        pred_label, max_support_prob, used_chunk, support_prob_per_chunk = [], [], [], []

        for idx in range(len(probs_per_doc_claim_pair)):

            doc = docs[idx]
            claim = claims[idx]

            # SentenceFusion: Reshape the probabilities into a matrix of shape (num_chunks x num_sentences)
            claim_sentences = self.split_into_sentences(claim)
            num_chunks = len(self.get_all_chunks_per_doc(doc, claim)['doc_chunks'])
            num_sentences = len(claim_sentences)
            prob_matrix = np.array(probs_per_doc_claim_pair[idx]).reshape(num_chunks, num_sentences)

            # For each sentence, pick the maximum probability across all chunks
            max_prob_per_sentence = np.max(prob_matrix, axis=0)

            # The final score is the minimum of these maximum values
            final_score = np.min(max_prob_per_sentence)

            pred_label.append(1 if final_score > 0.5 else 0)
            max_support_prob.append(final_score)
            used_chunk.append(self.get_all_chunks_per_doc(doc, claim)['doc_chunks'])
            support_prob_per_chunk.append(prob_matrix)

        return pred_label, max_support_prob, used_chunk, support_prob_per_chunk

    def split_into_sentences(self, text: str) -> List[str]:
        return nltk.sent_tokenize(text)