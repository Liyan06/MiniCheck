import json
import pandas as pd
import json
import time
import logging
from nltk.tokenize import sent_tokenize
from itertools import permutations
from prompt_utils import *
import numpy as np
import argparse


ATOMIC_FACT_GEN_PROMPT = """Document:
[DOCUMENT]

Please generate a summary for the document with the following requirements:
1. The summary should be a fluent and grammatical sentence.
2. The summary should be no more than 15 words.
3. The summary should cover information across the document.
Summary:"""


def split_into_chunks(text, chunk_num=3):

    sentences = sent_tokenize(text)

    try:
        num_sentences = len(sentences)
        chunk_size = num_sentences // chunk_num

        chunks = [sentences[i:i+chunk_size] for i in range(0, num_sentences, chunk_size)]
        chunked_text = [' '.join(chunk) for chunk in chunks]

        if len(chunked_text) > chunk_num:
            assert len(chunked_text) == chunk_num + 1
            chunked_text[-2] = chunked_text[-2] + " " + chunked_text[-1]
            chunked_text = chunked_text[:-1]
    except:
        return 'invalid'

    return chunked_text


def get_response_for_all_chunks(chunks):
    
    responses = []
    total_cost = 0
    for chunk in chunks:
        atomic_fact_gen_prompt_adapted = ATOMIC_FACT_GEN_PROMPT.replace("[DOCUMENT]", chunk)

        # better have a limit on the number of retries
        retry = True
        while retry:
            try:
                response, cost = get_GPT_output(atomic_fact_gen_prompt_adapted, return_cost=True, is_json=False)
                retry = False 
            except Exception as e:
                retry = True
                time.sleep(10)

        responses.append(response)
        total_cost += cost
    return responses, total_cost


def get_permutations_of_facts(facts: list):
    '''
    Generate all permutations of facts.
    key: indices of facts in permutation order
    value: list of facts in permutation order

    We only want combination of 2 sentences.
    '''

    assert type(facts) == list
    facts = {i+1: fact for i, fact in enumerate(facts)}

    all_permutations = []
    for r in range(1, len(facts) + 1):
        all_permutations.extend(permutations(facts, r))

    merge_idx_sents_dict = {}
    for permutation in all_permutations:
        if len(permutation) <= 2:
            permutation_str = [str(s) for s in permutation]
            fact_idx = "-".join(permutation_str)
            fact_to_merge = [facts[idx] for idx in permutation]
            merge_idx_sents_dict[fact_idx] = fact_to_merge
    
    return merge_idx_sents_dict


def leave_one_sent_out_chunk_construction(chunk, claim, atomic_facts_for_claim):
    '''
    Remove one sentence from the chunk so that the claim cannot be fully supported.

    Return a dictiaonary with edited_paragraph and atomic_fact_labels
    '''

    # construct passages with the i-th sentence removed
    # then keep the constructed passages that do not support the claim
    sentences = sent_tokenize(chunk)
    neg_paragraphs_for_claim = []

    costs = 0
    for i in range(len(sentences)):
        new_paragraph = [sentence for j, sentence in enumerate(sentences) if j != i]
        new_paragraph = ' '.join(new_paragraph)

        # we want some atomic fact not supported by the new paragraph
        atomic_facts_labels = []
        for atomic_fact in atomic_facts_for_claim:
            label, cost = entailment_check_for_document(claim=atomic_fact, document=new_paragraph, return_cost=True, is_json=False, n=3)
            atomic_facts_labels.append(label)
            costs += cost
        
        if 'No' in atomic_facts_labels:
            neg_paragraphs_for_claim.append((new_paragraph, atomic_facts_labels))

    return neg_paragraphs_for_claim, costs



class D2C_pipeline:

    def __init__(self):
        self.logging = logging.getLogger()
        self.total_cost = 0
        self.chunk_num = 3

    
    def construct_data(self, DOCUMENT):

        # Split a document into 3 chunks and get one summary/claim for each chunk
        CHUNKS, CLAIMS = d2c_pipeline.get_chunk_and_claims(DOCUMENT)

        # Claim Decomposition (Get atomic facts for each claim)
        CLAIM_ATOMIC_FACT_DICT = d2c_pipeline.get_atomic_facts_for_all_claims_in_doc(CLAIMS)
        
        # Sub-claim augmentation
        AUGMENTED_SENT_MAPPING = d2c_pipeline.claim_augmentations(CLAIMS)
        
        # Document-claim augmentation (Leave-one-out unsupporting chunks construction)
        EDITED_PARAGRAPHS = d2c_pipeline.document_claim_augmentation(CHUNKS, CLAIMS, CLAIM_ATOMIC_FACT_DICT)

        # Cross-document-claim augmentation
        REMAIN_CHUNKS_LABELS_FOR_ATOMIC_FACTS = d2c_pipeline.cross_document_claim_augmentation(CHUNKS, CLAIMS, CLAIM_ATOMIC_FACT_DICT, EDITED_PARAGRAPHS)


        # Construct claim-doc-label triples in a dataframe
        df = d2c_pipeline.construct_claim_doc_label_triples(CHUNKS, CLAIMS, CLAIM_ATOMIC_FACT_DICT, EDITED_PARAGRAPHS, REMAIN_CHUNKS_LABELS_FOR_ATOMIC_FACTS)

        return df


    def get_chunk_and_claims(self, text):

        self.logging.info(f"\n#### Split the document and obtain claims for chunks")

        step_cost = 0
        chunks = split_into_chunks(text, chunk_num=self.chunk_num)
        if chunks == 'invalid':
            CLAIMS = ['invalid']
        else:
            CLAIMS, cost = get_response_for_all_chunks(chunks)
            step_cost += cost

        self.total_cost += cost
        self.logging.info(f"Generated claims: {CLAIMS}")
        self.logging.info(f"Claim generation cost: {step_cost}")
        
        return chunks, CLAIMS


    def decompose_sent_to_facts(self, CLAIM, model="gpt-3.5-turbo-0125", return_cost=True, is_json=False):

        """
        Decompose a claim into atomic facts
        """

        prompt_for_decompose_adapted = PROMPT_FOR_DECOMPOSE.replace("[SENTENCE]", CLAIM)

        # better have a limit on the number of retries
        retry = True
        while retry:
            try:
                response, cost = get_GPT_output(prompt_for_decompose_adapted, model=model, return_cost=return_cost, is_json=is_json)
                retry = False 
            except Exception as e:
                retry = True
                time.sleep(10)
            
        ATOMIC_FACTS = [item[2:] if '- ' == item[:2] else item for item in response.split("\n")]

        self.logging.info(f"Atomic_facts: {ATOMIC_FACTS}")
        self.logging.info(f"Decomposition Cost: {cost}")

        return ATOMIC_FACTS, cost
    

    def get_atomic_facts_for_all_claims_in_doc(self, CLAIMS):

        """
        Get atomic facts for all claims in the document
        """

        self.logging.info(f"\n#### Decomposition Step")

        step_cost = 0
        CLAIM_ATOMIC_FACT_DICT = {}
        for CLAIM in CLAIMS:
            atomic_facts, cost = self.decompose_sent_to_facts(CLAIM)
            CLAIM_ATOMIC_FACT_DICT[CLAIM] = atomic_facts
            step_cost += cost
            self.logging.info(json.dumps({'Claim': CLAIM, 'Atomic Facts': atomic_facts}, indent=4))

        self.total_cost += step_cost
        self.logging.info(f"Atomic facts cost: {step_cost}")
        
        return CLAIM_ATOMIC_FACT_DICT
    

    def claim_augmentations(self, CLAIMS):

        """
        Claim augmentation
        """

        self.logging.info(f"\n#### Claim Augmentation Step")

        step_cost = 0
        merge_idx_sents_dict = get_permutations_of_facts(CLAIMS)
        AUGMENTED_SENT_MAPPING = {}
        for idx, facts in merge_idx_sents_dict.items():
            if len(facts) == 1:
                AUGMENTED_SENT_MAPPING[idx] = facts[0]
            else:
                response, cost = merge_facts_to_sent(facts)
                AUGMENTED_SENT_MAPPING[idx] = response
                step_cost += cost

        self.logging.info(f"Augmented Claims: {json.dumps(AUGMENTED_SENT_MAPPING, indent=4)}")
        self.logging.info(f"Claim augmentation cost: {step_cost}")

        return AUGMENTED_SENT_MAPPING
    

    def document_claim_augmentation(self, CHUNKS, CLAIMS, CLAIM_ATOMIC_FACT_DICT):

        """
        Document-claim augmentation
        """

        self.logging.info(f"\n#### Document-claim augmentation (Leave-one-out unsupporting chunks construction). This step may take a while.")

        step_cost = 0
        EDITED_PARAGRAPHS = {}

        if len(CHUNKS) == len(CLAIMS) == len(CLAIM_ATOMIC_FACT_DICT):
            
            for idx, (chunk, claim) in enumerate(zip(CHUNKS, CLAIMS)):
                atomic_facts = CLAIM_ATOMIC_FACT_DICT[claim]
                neg_paragraphs_for_claim, costs = leave_one_sent_out_chunk_construction(chunk=chunk, claim=claim, atomic_facts_for_claim=atomic_facts)
                step_cost += costs
                EDITED_PARAGRAPHS[str(idx+1)] = neg_paragraphs_for_claim
        
        self.total_cost += step_cost
        self.logging.info(f"Augmented paragraphs:\n{json.dumps(EDITED_PARAGRAPHS, indent=4)}")
        self.logging.info(f"Document-claim augmentation cost: {step_cost}")

        return EDITED_PARAGRAPHS
    

    def cross_document_claim_augmentation(self, CHUNKS, CLAIMS, CLAIM_ATOMIC_FACT_DICT, EDITED_PARAGRAPHS):

        """
        Cross-document claim augmentation
        """

        self.logging.info(f"\n#### Cross-document claim augmentation")

        step_cost = 0
        REMAIN_CHUNKS_LABELS_FOR_ATOMIC_FACTS = {}
        for sent_idx, items in EDITED_PARAGRAPHS.items():
            remaining_doc_chunks_ids = [i for i, chunk in enumerate(CHUNKS) if i != int(sent_idx)-1]
            
            sent_idx_remain_chunks_and_labels = {}
            for edited_paragraph, atomic_fact_labels in items:
                unsupported_claim = CLAIMS[int(sent_idx)-1]
                corresponding_atomic_facts = CLAIM_ATOMIC_FACT_DICT[unsupported_claim]

                remain_chunks_and_labels = []
                for remaining_doc_chunks_id in remaining_doc_chunks_ids:
                    remaining_doc_chunk = CHUNKS[remaining_doc_chunks_id]
                    labels = []
                    for corresponding_atomic_fact in corresponding_atomic_facts:
                        label, cost = entailment_check_for_document(claim=corresponding_atomic_fact, document=remaining_doc_chunk, return_cost=True, is_json=False, n=3)
                        labels.append(label)
                        step_cost += cost

                    remain_chunks_and_labels.append((str(remaining_doc_chunks_id + 1), labels))

                sent_idx_remain_chunks_and_labels[edited_paragraph] = remain_chunks_and_labels
            
            REMAIN_CHUNKS_LABELS_FOR_ATOMIC_FACTS[sent_idx] = sent_idx_remain_chunks_and_labels

        self.total_cost += step_cost
        self.logging.info(f"{json.dumps(REMAIN_CHUNKS_LABELS_FOR_ATOMIC_FACTS, indent=4)}")
        self.logging.info(f"Cross-document claim augmentation cost: {step_cost}")

        return REMAIN_CHUNKS_LABELS_FOR_ATOMIC_FACTS
    

    def construct_claim_doc_label_triples(self, CHUNKS, CLAIMS, CLAIM_ATOMIC_FACT_DICT, EDITED_PARAGRAPHS, REMAIN_CHUNKS_LABELS_FOR_ATOMIC_FACTS):

        """
        Construct claim-doc-label triples in a dataframe"""

        self.logging.info(f"\n#### Construct claim-doc-label triples in a dataframe")

        new_df = pd.DataFrame(columns=['claim', 'doc', 'label'])

        for claim_idx, claim in enumerate(CLAIMS):
            new_df.loc[len(new_df)] = [claim, CHUNKS[claim_idx], 1]

        for sent_idx, items in EDITED_PARAGRAPHS.items():

            claim = CLAIMS[int(sent_idx)-1]
            corresponding_atomic_facts = CLAIM_ATOMIC_FACT_DICT[claim]
            check_claim_by_rest_para_dict = REMAIN_CHUNKS_LABELS_FOR_ATOMIC_FACTS[sent_idx]

            for edited_paragraph, atomic_fact_labels in items:

                for atomic_idx, atomic_fact in enumerate(corresponding_atomic_facts):
                    if atomic_fact_labels[atomic_idx] == 'Yes':
                        new_df.loc[len(new_df)] = [atomic_fact, edited_paragraph, 1]
                    elif atomic_fact_labels[atomic_idx] == 'No':
                        new_df.loc[len(new_df)] = [atomic_fact, edited_paragraph, 0]
                    else:
                        raise ValueError("The atomic fact label is not 'Yes' or 'No'")
                    
                if 'No' in atomic_fact_labels:
                    wrong_fact = str([corresponding_atomic_facts[i] for i in np.where(np.array(atomic_fact_labels) == 'No')[0]])
                    new_df.loc[len(new_df)] = [claim, edited_paragraph, 0]
                else:
                    new_df.loc[len(new_df)] = [claim, edited_paragraph, 1]


            if len(check_claim_by_rest_para_dict.values()) != 0:
                for item in list(check_claim_by_rest_para_dict.values())[0]:
                    para_idx = int(item[0]) - 1

                    for atomic_idx, atomic_fact in enumerate(corresponding_atomic_facts):
                        if item[1][atomic_idx] == 'Yes':
                            new_df.loc[len(new_df)] = [atomic_fact, CHUNKS[para_idx], 1]
                        elif item[1][atomic_idx] == 'No':
                            new_df.loc[len(new_df)] = [atomic_fact, CHUNKS[para_idx], 0]
                        else:
                            raise ValueError("The atomic fact label is not 'Yes' or 'No")
                        
                    if 'No' in item[1]:
                        wrong_fact = str([corresponding_atomic_facts[i] for i in np.where(np.array(item[1]) == 'No')[0]])
                        new_df.loc[len(new_df)] = [claim, CHUNKS[para_idx], 0]
                    else:
                        new_df.loc[len(new_df)] = [claim, CHUNKS[para_idx], 1]
        new_df = new_df[new_df.doc.apply(lambda x: len(x)) >= 500].reset_index(drop=True)

        self.logging.info(f"Constructed dataframe:\n{new_df}")

        return new_df



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_path', type=str, default='D2C-doc-example.txt', help='path to the document that will be used to construct the training data.')
    parser.add_argument('--no_log', action='store_true', help='Disable logging')
    args = parser.parse_args()

    if args.no_log:
        logging.basicConfig(level=logging.CRITICAL)
    else:
        logging.basicConfig(
            level=logging.INFO,  
            format='%(message)s',
            handlers=[logging.StreamHandler()]
        )
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    with open(args.doc_path) as file:
        DOCUMENT = file.read()

    d2c_pipeline = D2C_pipeline()
    claim_doc_label_df = d2c_pipeline.construct_data(DOCUMENT)

    logging.info(f"\n#### Total Cost: {d2c_pipeline.total_cost}")