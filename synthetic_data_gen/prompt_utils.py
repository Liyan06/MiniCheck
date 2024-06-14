import re
import time
from itertools import combinations
import json
from openai import OpenAI


with open("/path/to/your/key.json") as file:
    data = json.load(file) 
    OPENAI_API_KEY = data['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_API_KEY)


PROMPT_FOR_DECOMPOSE = """Segment the following sentence into individual facts:

Sentence: Other title changes included Lord Steven Regal and The Nasty Boys winning the World Television Championship and the World Tag Team Championship respectively.
Facts:
- Lord Steven Regal won the World Television Championship. 
- The Nasty Boys won and the World Tag Team Championship.

Sentence: The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.
Facts:
- The parkway was opened in 2001.
- The parkway was opened after just under a year of construction.
- The parkway was opened after two decades of community requests.

Sentence: Touring began in Europe in April-June with guitarist Paul Gilbert as the opening act, followed by Australia and New Zealand in July, Mexico and South America in late July-August, and concluding in North America in October-November.
Facts:
- Touring began in Europe in April-June.
- The opening act of the tour was guitarist Paul Gilbert.
- The tour was in Australia and New Zealand in July.
- The tour was in Mexico and South America in late July-August.
- The tour was concluded in North America in October-November.

Sentence: In March 2018, the company partnered With Amazon Web Services (AWS) to offer Al-enabled conversational solutions to customers in India.
Facts:
- The company partnered with Amazon Web Services (AWS) in March 2018.
- The two companies partnered to offer Al-enabled conversational solutions to customers in India.

Sentence: The most significant of these is in Germany, which now has a Yazidi community of more than 200,000 living primarily in Hannover, Bielefeld, Celle, Bremen, Bad Oeynhausen, Pforzheim and Oldenburg.
Facts:
- The most significant of these is in Germany.
- Germany now has a Yazidi community of more than 200,000.
- Yazidi community in Germany lives primarily in Hannover, Bielefeld, Celle, Bremen, Bad Oeynhausen, Pforzheim and Oldenburg.

Sentence: A previous six-time winner of the Nations' Cup, Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, who made the final for the fourth time, 2-0.
Facts:
- Sebastian Vettel is a previous six-time winner of the Nations' Cup.
- Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, 2-0.
- Tom Kristensen made the final for the fourth time.

Sentence: [SENTENCE]
Facts:\n"""


DEDUCTION_GEN_PROMPT_V2 = """Your task is to generate a pair of sentences so that the provided claim can be entailed by the sentence pair. You must make sure that the claim can only be deduced by combining the information from the two sentences that contain unique information.

Examples:
Provided Claim: The investigation is into allegations that his mayoral campaign received illegal foreign funds.
Sentence 1: During the period leading up to the mayoral election, there was a notable increase in his campaign's financial resources.
Sentence 2: Investigation shows the funds having origins beyond national boundaries, a detail raising questions under current campaign laws.

Provided Claim: Approximately 1,000 fans fainted at the concert.
Sentence 1: Emergency services reported an unusually high number of calls for medical assistance during the concert with an attendance of 20,000.
Sentence 2: Venue officials estimated that approximately 5% of the audience required medical attention for fainting.

Provided Claim: The interest rate hikes were intended to manage inflation and moderate economic growth.
Sentence 1: Central bank officials expressed concern over the rising consumer price index and the overheating of the economy.
Sentence 2: The monetary policy committee decided to adjust the interest rates as a response to these economic indicators.

Provided Claim: Several advertisers are considering halting their ads on social media platform X.
Sentence 1: Some companies are re-evaluating their marketing strategies to avoid association with platforms that fail to address misinformation.
Sentence 2: Recent reports show that platform X has received criticism for its handling of false information spreading unchecked.

Please make sure that NEITHER sentence alone supports the claim.

Your turn:
Provided Claim: [CLAIM]"""



DOCUMENT_GEN_PROMPT = """We are creating a news article (one paragraph) in the style of The New York Times. We will give you a list of facts to use when writing your article. You must include all the facts in the list. Never state deduced facts or conclusions. The article should stick to the fact list pretty closely. Include as many sentences as needed to write each fact from the list of facts.

Facts you must include:
[FACT-STRING]

Output:\n"""


PROMPT_FOR_MERGE = """Merge the following individual facts into a single sentence:

Facts:
- Lord Steven Regal wan the World Television Championship. 
- The Nasty Boys wan and the World Tag Team Championship.
Sentence: Other title changes included Lord Steven Regal and The Nasty Boys winning the World Television Championship and the World Tag Team Championship respectively.

Facts:
- The parkway was opened in 2001.
- The parkway was opened after just under a year of construction.
- The parkway was opened after two decades of community requests.
Sentence: The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.

Facts:
- Touring began in Europe in April-June.
- The opening act was guitarist Paul Gilbert.
- There was a tour in Australia in July.
- There was a tour in New Zealand in July.
- There was a tour in Mexico in late July-August.
- There was a tour in South America in late July-August
- The tour was concluded in North America in October-November.
Sentence: Touring began in Europe in April-June with guitarist Paul Gilbert as the opening act, followed by Australia and New Zealand in July, Mexico and South America in late July-August, and concluding in North America in October-November.

Facts:
- The company partnered with Amazon Web Services (AWS) in March 2018.
- The two companies partnered to offer Al-enabled conversational solutions to customers in India.
Sentence: In March 2018, the company partnered With Amazon Web Services (AWS) to offer Al-enabled conversational solutions to customers in India.

Facts:
- The most significant of these is in Germany.
- Germany now has a Yazidi community of more than 200,000.
- Yazidi community in Germany lives primarily in Hannover.
- Yazidi community in Germany lives primarily in Bielefeld.
- Yazidi community in Germany lives primarily in Celle.
- Yazidi community in Germany lives primarily in Bremen.
- Yazidi community in Germany lives primarily in Bad Oeynhausen.
- Yazidi community in Germany lives primarily in Pforzheim.
- Yazidi community in Germany lives primarily in Oldenburg.
Sentence: The most significant of these is in Germany, which now has a Yazidi community of more than 200,000 living primarily in Hannover, Bielefeld, Celle, Bremen, Bad Oeynhausen, Pforzheim and Oldenburg.

Facts:
- Sebastian Vettel is a previous six-time winner of the Nations' Cup.
- Sebastian Vettel became Champion of Champions for the first time.
- Sebastian Vettel defeated Tom Kristensen.
- Tom Kristensen made the final for the fourth time.
- The score was 2-0.
Sentence: A previous six-time winner of the Nations' Cup, Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, who made the final for the fourth time, 2-0.

Facts:
[FACTS]
Sentence:"""


ENTAILMENT_CHECK_PROMPT = """Source: [SOURCE]
Claim: [CLAIM]

Is the claim fully entailed, or implied, by the source? Please answer with "yes" or "no"."""


ENTAILMENT_CHECK_PROMPT_NO_PRIOR_KNOWLEDGE = """Source: [SOURCE]
Claim: [CLAIM]

Is the claim entailed, or implied, by the source? Please do not rely on your knowledge about the person or event mentioned in the claim. Please answer with "yes" or "no"."""


def get_GPT_output(prompt, temperature=1, model="gpt-4-0125-preview", return_cost=False, is_json=False, n=1):

    kwargs = {}

    if is_json:
        kwargs["response_format"] = { "type": "json_object"}
        
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        n=n,
        **kwargs
    )
    answer = response.choices[0].message.content

    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    
    if return_cost:
        total_usd = 0.0

        inp_token_cost, out_token_cost = 0.0, 0.0

        if model == "gpt-3.5-turbo-0125":
            inp_token_cost, out_token_cost = 0.0005, 0.0015
        elif model == 'gpt-4-0125-preview':
            inp_token_cost, out_token_cost = 0.01, 0.03
        else:
            raise ValueError("Invalid model")
        total_usd = (prompt_tokens / 1000) * inp_token_cost + (completion_tokens / 1000) * out_token_cost

        if n == 1:
            return answer, total_usd
        else:
            return response.choices, total_usd

    else:
        if n == 1:
            return answer
        else:
            return response.choices


def generate_deduction_pair(claim, return_cost=True, is_json=False):

    """
    Generate a pair of sentences so that the provided claim can be entailed by the sentence pair.
    """

    deduction_gen_prompt_adapted = DEDUCTION_GEN_PROMPT_V2.replace("[CLAIM]", claim)

    retry = True
    counter = 2
    while retry and counter >= 0:
        try:
            response, cost = get_GPT_output(deduction_gen_prompt_adapted, return_cost=return_cost, is_json=is_json)
            
            sentences = response.split("\n")
            sentences = [sent for sent in sentences if sent != ''][:2]
            if len(sentences) != 2:
                counter -= 1
                raise Exception("The model should generate exactly two sentences.")
            sentences = [re.sub(r'^Sentence \d+: ', '', sent) for sent in sentences]
                        
            retry = False 
        except Exception as e:
            retry = True
            time.sleep(10)

    return (sentences[0], sentences[1]), cost


def generate_document(pairs_of_sents: list, return_cost=True, is_json=False):

    """
    Generate a news article (one paragraph) in the style of The New York Times using the sentences from sentence pairs.
    """

    facts_string = ''
    for sent in pairs_of_sents:
        facts_string += '- ' + sent + '\n'
    document_gen_prompt_adapted = DOCUMENT_GEN_PROMPT.replace('[FACT-STRING]', facts_string)

    retry = True
    while retry:
        try:
            response, cost = get_GPT_output(document_gen_prompt_adapted, return_cost=return_cost, is_json=is_json)
            retry = False 
        except Exception as e:
            retry = True
            time.sleep(10)

    return response, cost


def get_combinations_of_facts(facts: list):

    '''
    # Generate all combinations of facts
    key: indices of facts to merge
    value: list of facts to merge
    '''

    assert type(facts) == list
    facts = {i+1:fact for i, fact in enumerate(facts)}

    all_combinations = []
    for r in range(1, len(facts) + 1):
        all_combinations.extend(combinations(facts, r))
    all_combinations

    merge_idx_sents_dict = {}
    for combination in all_combinations:
        combination_str = [str(s) for s in combination]
        fact_idx = "-".join(combination_str)
        fact_to_merge = [facts[idx] for idx in combination]
        merge_idx_sents_dict[fact_idx] = fact_to_merge
    
    return merge_idx_sents_dict


def merge_facts_to_sent(facts: list, model="gpt-3.5-turbo-0125", return_cost=True, is_json=False):

    facts_string = ''
    for sent in facts:
        facts_string += '- ' + sent + '\n'
    facts_string = facts_string.strip()
    prompt_for_merge_adapted = PROMPT_FOR_MERGE.replace("[FACTS]", facts_string)

    retry = True
    while retry:
        try:
            response, cost = get_GPT_output(prompt_for_merge_adapted, model=model, return_cost=return_cost, is_json=is_json)
            retry = False 
        except Exception as e:
            retry = True
            time.sleep(10)

    return response, cost


def construct_removed_sent_dict(fact_sent_dict):

    '''
    Create a dictionary that can be used for generating 
    passages (with specific sentences removed) and a dictionary 
    for checking a fact (one sentence removed) against a set of 
    remaining facts + the sentence not removed from the sentence pair.
    '''

    sent_dict_for_passage_gen = {}
    atomic_dict_for_fact_check = {}

    for fact_idx, (fact_key, sentences) in enumerate(fact_sent_dict.items()):

        for i, sentence_to_remove in enumerate(sentences):
            new_key = f"r-fact-{fact_idx + 1}-sent-{i + 1}"

            remaining_sentences = []
            facts_as_resource = []

            # Iterate through all facts and their sentences to collect the remaining sentences
            for other_fact, other_sentences in fact_sent_dict.items():
                if other_fact != fact_key:
                    remaining_sentences.extend(other_sentences)
                    facts_as_resource.append(other_fact)
                else:
                    # Add all sentences except the one to be removed
                    remaining_sentences.extend([s for s in other_sentences if s != sentence_to_remove])
                    facts_as_resource.extend([s for s in other_sentences if s != sentence_to_remove])

            sent_dict_for_passage_gen[new_key] = remaining_sentences
            atomic_dict_for_fact_check[new_key] = (facts_as_resource, fact_key)

    return sent_dict_for_passage_gen, atomic_dict_for_fact_check


def entailment_check_for_sent_pair(claim, sent_pair: list, return_cost=True, is_json=False, n=1):
    assert len(sent_pair) == 2
    sent_pair_formatted = " ".join(sent_pair)
    entailment_check_prompt_adapted = ENTAILMENT_CHECK_PROMPT.replace("[SOURCE]", sent_pair_formatted).replace("[CLAIM]", claim)

    retry = True
    while retry:
        try:
            response, cost = get_GPT_output(entailment_check_prompt_adapted, return_cost=return_cost, is_json=is_json, n=n)
            if n > 1:
                response = [res.message.content for res in response]
                response = ['Yes' if 'yes' in res.lower() else 'No' for res in response]
                response = max(set(response), key=response.count)
            retry = False 
        except Exception as e:
            retry = True
            time.sleep(10)

    return response, cost


def entailment_check_for_document(claim, document, return_cost=True, is_json=False, n=1):
    entailment_check_prompt_adapted = ENTAILMENT_CHECK_PROMPT.replace("[SOURCE]", document).replace("[CLAIM]", claim)

    retry = True
    while retry:
        try:
            response, cost = get_GPT_output(entailment_check_prompt_adapted, return_cost=return_cost, is_json=is_json, n=n)
            if n > 1:
                response = [res.message.content for res in response]
                response = ['Yes' if 'yes' in res.lower() else 'No' for res in response]
                response = max(set(response), key=response.count)
            retry = False 
        except Exception as e:
            retry = True
            time.sleep(10)

    return response, cost


def entailment_check_for_claim(claim, source, return_cost=True, is_json=False, n=1):
    entailment_check_prompt_adapted = ENTAILMENT_CHECK_PROMPT_NO_PRIOR_KNOWLEDGE.replace("[SOURCE]", source).replace("[CLAIM]", claim)

    retry = True
    while retry:
        try:
            response, cost = get_GPT_output(entailment_check_prompt_adapted, return_cost=return_cost, is_json=is_json, n=n)
            if n > 1:
                response = [res.message.content for res in response]
                response = ['Yes' if 'yes' in res.lower() else 'No' for res in response]
                response = max(set(response), key=response.count)
            retry = False 
        except Exception as e:
            retry = True
            time.sleep(10)

    return response, cost




