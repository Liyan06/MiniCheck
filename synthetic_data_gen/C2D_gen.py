from prompt_utils import *
import json
import argparse
import pandas as pd
import logging


def construct_removed_sent_dict_wrapper(ATOMIC_FACTS, SENT_PAIRS):

    '''
    Create a dictionary that can be used for generating 
    passages (with specific sentences removed) and a dictionary 
    for checking a fact (one sentence removed) against a set of 
    remaining facts + the sentence not removed from the sentence pair.
    '''

    atom_sent_map = {}
    for atom, sent_pair in zip(ATOMIC_FACTS, SENT_PAIRS):
        atom_sent_map[atom] = sent_pair

    sent_dict_for_passage_gen, atomic_dict_for_fact_check = construct_removed_sent_dict(atom_sent_map)    
    
    return sent_dict_for_passage_gen, atomic_dict_for_fact_check


class C2D_pipeline:

    def __init__(self):
        self.logging = logging.getLogger()
        self.total_cost = 0

    def construct_data(self, CLAIM):

        self.logging.info(f"\n#### Claim for training data construction: {CLAIM}")

        ATOMIC_FACTS = self.decompose_sent_to_facts(CLAIM)

        # Generate sentence pairs for atomic facts, followed by GPT-4 checking
        SENT_PAIRS, SENT_PAIRS_LABELS = self.sent_pairs_generation(ATOMIC_FACTS)

        # Generate a passage for the original claim
        ORG_PASSAGE, ORG_PASSAGE_LABEL = self.org_passage_generation(SENT_PAIRS)

        # Augment/Merge claims by recombining atomic facts ONLY
        # Guarantee those claims are supported by the original passage    
        AUGMENTED_ATOMIC_FACTS = self.augment_atomic_facts(CLAIM, ATOMIC_FACTS)

        # Construct a dictionary that can be used for generating passages (with specific sentences removed) and a dictionary for checking a fact (one sentence removed) against a set of remaining facts + the sentence not removed from the sentence pair
        sent_dict_for_passage_gen, atomic_dict_for_fact_check = construct_removed_sent_dict_wrapper(ATOMIC_FACTS, SENT_PAIRS)

        # Check if a set of remaining facts + the sentence not removed from the sentence pair supports the atomic fact with one of its sentence from the sentence pair removed
        ATOMIC_FACT_LABEL_WITH_SENT_REMOVAL = self.get_atomic_fact_label_after_sent_removal(atomic_dict_for_fact_check)

        # Passage Augmentation
        AUGMENTED_PASSAGES = self.passage_augmentation(SENT_PAIRS, ATOMIC_FACT_LABEL_WITH_SENT_REMOVAL, sent_dict_for_passage_gen)

        # Construct the dataset
        claim_doc_label_df = self.construct_claim_doc_label_triples(CLAIM, SENT_PAIRS, SENT_PAIRS_LABELS, ORG_PASSAGE, ORG_PASSAGE_LABEL, AUGMENTED_PASSAGES, AUGMENTED_ATOMIC_FACTS, ATOMIC_FACT_LABEL_WITH_SENT_REMOVAL)

        return claim_doc_label_df


    def decompose_sent_to_facts(self, CLAIM, model="gpt-3.5-turbo-0125", return_cost=True, is_json=False):

        """
        Decompose a claim into atomic facts
        """
        self.logging.info(f"\n#### Decomposition Step")

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

        self.total_cost += cost
        self.logging.info(f"Atomic_facts: {ATOMIC_FACTS}")
        self.logging.info(f"Decomposition Cost: {cost}")

        return ATOMIC_FACTS
    

    def sent_pairs_generation(self, ATOMIC_FACTS):

        """
        Generate sentence pairs for atomic facts, followed by GPT-4 checking
        """
        self.logging.info(f"\n#### Sentence Pair Generation Step")

        SENT_PAIRS = []
        SENT_PAIRS_LABELS = []

        step_cost = 0
        for atomic_fact in ATOMIC_FACTS:

            is_not_entailed = True
            counter = 5
            while is_not_entailed and counter > 0:
                sent_pair, cost_decuct = generate_deduction_pair(atomic_fact)
                sent_pair_label, cost_entail = entailment_check_for_sent_pair(atomic_fact, sent_pair, n=3)
                step_cost += cost_decuct + cost_entail
                if 'yes' in sent_pair_label.lower(): # yes: the sentence pair supports the atomic fact
                    is_not_entailed = False
                else:
                    counter -= 1

            SENT_PAIRS.append(sent_pair)
            SENT_PAIRS_LABELS.append(sent_pair_label)
        
        self.total_cost += step_cost
        self.logging.info(f"Sent pairs: {SENT_PAIRS}")
        self.logging.info(f"Sent pair labels: {SENT_PAIRS_LABELS}")
        self.logging.info(f"Sent pair generation cost: {step_cost}")

        return SENT_PAIRS, SENT_PAIRS_LABELS
    

    def org_passage_generation(self, SENT_PAIRS):

        """
        Generate a passage for the original claim
        """
        self.logging.info(f"\n#### Original Passage Generation Step")

        step_cost = 0
        sents_from_sent_pairs = []
        for sent_pair in SENT_PAIRS: 
            sents_from_sent_pairs.extend(sent_pair)

        is_not_entailed = True
        counter = 5
        while is_not_entailed and counter > 0:

            ORG_PASSAGE, cost = generate_document(sents_from_sent_pairs, return_cost=True, is_json=False)
            step_cost += cost

            num_passed_fact = 0
            sents_in_pair_group = [sents_from_sent_pairs[i:i+2] for i in range(0, len(sents_from_sent_pairs), 2)]

            for sents_in_pair in sents_in_pair_group:
                sent = " ".join(sents_in_pair)
                ORG_PASSAGE_LABEL, cost_entail = entailment_check_for_document(sent, ORG_PASSAGE, n=3)
                step_cost += cost_entail
                if 'yes' not in ORG_PASSAGE_LABEL.lower():
                    is_not_entailed = True
                    counter -= 1
                    self.logging.info(f"Failing to generate a supporting document. Remaining attempts: {counter}")
                    break
                else:
                    num_passed_fact += 1
            if num_passed_fact == len(sents_in_pair_group):
                is_not_entailed = False

        self.total_cost += step_cost
        self.logging.info(f"Original Passage: {ORG_PASSAGE}")
        self.logging.info(f"Passage label: {ORG_PASSAGE_LABEL}")
        self.logging.info(f"Original passage generation cost: {step_cost}")  

        return ORG_PASSAGE, ORG_PASSAGE_LABEL  
    

    def augment_atomic_facts(self, CLAIM, ATOMIC_FACTS):

        """
        Augment/Merge claims by recombining atomic facts ONLY

        E.x.
        Augmented automic facts: {
            "1": "Over 5,000 members of the caravan were staying at the Tijuana Stadium by a certain date.",
            "2": "The Tijuana Stadium has a capacity of 3,000.",
            "1-2": "By this date, over 5,000 members of the caravan were staying at the Tijuana Stadium \u2014 a structure with a capacity of 3,000."
        }
        """
        self.logging.info(f"\n#### Atomic Facts Augmentation Step")

        step_cost = 0
        AUGMENTED_ATOMIC_FACTS = {}   
        merge_idx_sents_dict = get_combinations_of_facts(ATOMIC_FACTS)
        for idx, facts in merge_idx_sents_dict.items():
            if len(facts) == 1:
                # this suggests an atomic fact
                AUGMENTED_ATOMIC_FACTS[idx] = facts[0]
            elif len(facts) == len(ATOMIC_FACTS):
                # this suggests the original claim to be decomposed
                AUGMENTED_ATOMIC_FACTS[idx] = CLAIM
            else:
                response, cost = merge_facts_to_sent(facts)
                AUGMENTED_ATOMIC_FACTS[idx] = response
                step_cost += cost
        
        self.total_cost += step_cost
        self.logging.info(f"Augmented automic facts: {json.dumps(AUGMENTED_ATOMIC_FACTS, indent=4)}")

        return AUGMENTED_ATOMIC_FACTS
    
    def get_atomic_fact_label_after_sent_removal(self, atomic_dict_for_fact_check):

        """
        Check if a set of remaining facts + the sentence not removed from the sentence pair supports the atomic fact with one of its sentence from the sentence pair removed

        - ATOMIC_FACT_LABEL_WITH_SENT_REMOVAL:
        If label is False, that mean removing the sentence lead to the fact 
        unsupported, we will generate a document for this case

        E.x.
        {"r-fact-1-sent-1": true, "r-fact-1-sent-2": false, "r-fact-2-sent-1": true, "r-fact-2-sent-2": false}
        """
        self.logging.info(f"\n#### Fact Check Step (for sent removal)")

        step_cost = 0
        ATOMIC_FACT_LABEL_WITH_SENT_REMOVAL = {}
        for key, (source, fact) in atomic_dict_for_fact_check.items():
            source_sent = " ".join(source).strip()
            response, cost = entailment_check_for_claim(fact, source_sent, n=3)
            ATOMIC_FACT_LABEL_WITH_SENT_REMOVAL[key] = 'yes' in response.lower()
            step_cost += cost
        
        self.total_cost += step_cost
        self.logging.info(f"Fact check labels (for sent removal): {json.dumps(ATOMIC_FACT_LABEL_WITH_SENT_REMOVAL)}")
        self.logging.info(f"Fact check cost (for sent removal): {step_cost}")

        return ATOMIC_FACT_LABEL_WITH_SENT_REMOVAL
    

    def passage_augmentation(self, SENT_PAIRS, ATOMIC_FACT_LABEL_WITH_SENT_REMOVAL, sent_dict_for_passage_gen):

        """
        Generate augmented passages by removing one sentence from the sentence pair
        (1) Only generate a passage if removing a sentence cause the atomic fact not supported
        (2) Keep passages that inclues all prompted facts;

        E.x.
        Augmented passages: {
            "r-fact-1-sent-1": "...",
            "r-fact-1-sent-2": "...",
            "r-fact-2-sent-1": "...",
            "r-fact-2-sent-2": "..."
        }
        """
        self.logging.info(f"\n#### Passage Augmentation Step")

        step_cost = 0
        AUGMENTED_PASSAGES = {}
        for name, label in ATOMIC_FACT_LABEL_WITH_SENT_REMOVAL.items():
            # if label is False, that mean removing the sentence lead to the fact 
            # unsupported, we will generate a document for this case
            if label == False:

                is_not_entailed = True
                counter = 5
                while is_not_entailed and counter > 0:

                    document, cost = generate_document(sent_dict_for_passage_gen[name], return_cost=True, is_json=False)
                    step_cost += cost

                    atomic_fact_idx = int(name.split('-')[2]) - 1
                    sent_idx = int(name.split('-')[4]) - 1

                    remaining_sents_from_atomic_fact = [sent for i, sent in enumerate(SENT_PAIRS[atomic_fact_idx]) if i != sent_idx]
                    remaining_sent_pairs_reformatted = [" ".join(sent) for i, sent in enumerate(SENT_PAIRS) if i != atomic_fact_idx]
                    sents_to_check = remaining_sents_from_atomic_fact + remaining_sent_pairs_reformatted

                    num_passed_fact = 0
                    for sent in sents_to_check:
                        label, cost_entail = entailment_check_for_document(sent, document, n=3)
                        step_cost += cost_entail
                        if 'yes' not in label.lower():
                            is_not_entailed = True
                            counter -= 1
                            self.logging.info(f"Regenerate agumented passages. Remaining attempts: {counter}")
                            break
                        else:
                            num_passed_fact += 1
                    if num_passed_fact == len(sents_to_check):
                        is_not_entailed = False
                
                if is_not_entailed:
                    AUGMENTED_PASSAGES[name] = 'invalid_doc'
                else:
                    AUGMENTED_PASSAGES[name] = document
            
            # if label is True, that mean removing the sentence does not lead to the fact 
            # unsupported, we will not generate a document for this case
            else:
                AUGMENTED_PASSAGES[name] = 'invalid_doc'

        self.total_cost += step_cost
        self.logging.info(f"Augmented passages: {json.dumps(AUGMENTED_PASSAGES, indent=4)}")
        self.logging.info(f"Passage augmentation cost: {step_cost}")

        return AUGMENTED_PASSAGES
    

    def construct_claim_doc_label_triples(self, CLAIM, SENT_PAIRS, SENT_PAIRS_LABELS, ORG_PASSAGE, ORG_PASSAGE_LABEL, AUGMENTED_PASSAGES, AUGMENTED_ATOMIC_FACTS, FACT_LABEL_WITH_SENT_REMOVAL):
        self.logging.info(f"\n#### Construct claim-doc-label triples in a dataframe")

        # Construct the dataset
        df = pd.DataFrame(columns=['claim', 'sent_pair', 'sent_pair_label', 'org_passage', 'org_passage_label', 'augmented_passage', 'augmented_sent', 'fact_check_label'])
        df.loc[len(df)] = [CLAIM, SENT_PAIRS, SENT_PAIRS_LABELS, ORG_PASSAGE, ORG_PASSAGE_LABEL, AUGMENTED_PASSAGES, AUGMENTED_ATOMIC_FACTS, FACT_LABEL_WITH_SENT_REMOVAL]

        new_df = pd.DataFrame(columns=['claim', 'doc', 'label'])

        # First add data with valid sentence pair and passages
        # valid sentence pair: the pair can support the atomic fact
        # valid passage: the passage includes the sentence pair
        mask1 = df.sent_pair_label.apply(lambda x: all([label == 'Yes' for label in x]))
        mask2 = df.org_passage_label.apply(lambda x: x == 'Yes')
        df_valid = df[mask1 & mask2].reset_index(drop=True)

        for row, data in df_valid.iterrows():

            claim = data.claim
            fact_check_labels = data.fact_check_label
            augmented_sents = data.augmented_sent
            augmented_passages = data.augmented_passage
            org_passage = data.org_passage

            for augmented_sent in augmented_sents.values():
                new_df.loc[len(new_df)] = [augmented_sent, org_passage, 1]

            for remove_id, passage in augmented_passages.items():
                # remove_id: 'r-fact-i-sent-j'
                # this passage is constructed by removing the sent-j from the 
                # sentence pair of the atomic fact i
                atomic_fact_idx = remove_id.split('-')[2]

                # if the passage is invalid as the following, then we skip the document.
                # (1) the passage still support the fact-i after sentence-j from fact-i is removed; 
                # (2) the passage does not contain the required facts to be included.
                if passage != 'invalid_doc' and fact_check_labels[remove_id] != True:

                    # check if the atomic fact-i is supported after removing sentence j from its sentence pair
                    # if the fact label is False, then the atomic fact is unsupported by removing sentence j from atomic fact i
                    # This means any augmented sentence with fact-i is not supported by the passage.
                    fact_is_supported = fact_check_labels[remove_id]
                    if not fact_is_supported:

                        supported_claims = [value for key, value in augmented_sents.items() if atomic_fact_idx not in key]
                        unsupported_claims = [value for key, value in augmented_sents.items() if atomic_fact_idx in key]

                        for supported_claim in supported_claims:
                            new_df.loc[len(new_df)] = [supported_claim, passage, 1]
                        for unsupported_claim in unsupported_claims:
                            new_df.loc[len(new_df)] = [unsupported_claim, passage, 0]

        # The passage cannot support the original claim by construction
        df_invalid = df[~(mask1 & mask2)].reset_index(drop=True)
        for row, data in df_invalid.iterrows():
            unsupported_claim = data.claim
            passage = data.org_passage
            new_df.loc[len(new_df)] = [unsupported_claim, passage, 0]

        self.logging.info(f"Constructed dataframe:\n{new_df}")

        return new_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--claim', type=str, default='By this date, over 5,000 members of the caravan were staying at the Tijuana Stadium — a structure with a capacity of 3,000.', help='Claim used to construct the training data')
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

    c2d_pipeline = C2D_pipeline()
    claim_doc_label_df = c2d_pipeline.construct_data(args.claim)

    logging.info(f"\n#### Total Cost: {c2d_pipeline.total_cost}")
