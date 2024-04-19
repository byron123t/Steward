"""
- "annotation_id" (str): unique id for each task
- "website" (str): website name
- "domain" (str): website domain
- "subdomain" (str): website subdomain
- "confirmed_task" (str): task description
- "action_reprs" (list[str]): human readable string representation of the action sequence
- "actions" (list[dict]): list of actions (steps) to complete the task
    - "action_uid" (str): unique id for each action (step)
    - "raw_html" (str): raw html of the page before the action is performed
    - "cleaned_html" (str): cleaned html of the page before the action is performed
    - "operation" (dict): operation to perform
        - "op" (str): operation type, one of CLICK, TYPE, SELECT
        - "original_op" (str): original operation type, contain additional HOVER and ENTER that are mapped to CLICK, not used
        - "value" (str): optional value for the operation, e.g., text to type, option to select
    - "pos_candidates" (list[dict]): ground truth elements. Here we only include positive elements that exist in "cleaned_html" after our preprocessing, so "pos_candidates" might be empty. The original labeled element can always be found in the "raw_html".
        - "tag" (str): tag of the element
        - "is_original_target" (bool): whether the element is the original target labeled by the annotator
        - "is_top_level_target" (bool): whether the element is a top level target find by our algorithm. please see the paper for more details.
        - "backend_node_id" (str): unique id for the element
        - "attributes" (str): serialized attributes of the element, use `json.loads` to convert back to dict
    - "neg_candidates" (list[dict]): other candidate elements in the page after preprocessing, has similar structure as "pos_candidates"
    
    
Desired format (JSON)
- instruction
- input
- output
"""


import json, os, re, signal, argparse, tiktoken, base64, cv2
import pandas as pd
from tqdm import tqdm
from random import random, shuffle, randrange, choice
from difflib import get_close_matches
from src.html_processing import clean_element
from src.parse_html import ParseHtml
from src.smart_runtime import PageStateActor
from src.API import OpenAIAPI
from sensitive import *
from src.benchmark import BM


def interrupt_handler(signum, frame):
    global EXIT_BOOL
    print("\nInterrupt signal received. Finishing up and terminating...")
    EXIT_BOOL = True


def shuffle_list_by_indices(lst):
    shuffled_indices = list(range(len(lst)))
    shuffle(shuffled_indices)
    shuffled_list = [lst[i] for i in shuffled_indices]
    return shuffled_indices, shuffled_list


def filter_elements(actor, interactables, true_element, limit_elements=True):
    BM.mark('filter_elements')
    flattened_interactables_map = {}
    cleaned_elements_map = {}
    all_elements = {}
    flattened_interactables = []
    elements = []
    cleaned_elements = []
    sublists = []
    current_sublist = []
    current_tokens = 0
    counter = 0
    for tag in interactables:
        counter += 1
    print('TOTAL ELEMENTS: {}'.format(counter))
    for tag in interactables:
        try:
            element = tag.prettify()
            if element not in all_elements:
                all_elements[element] = 1
                flattened_interactables.append(tag)
                stripped_lowered_element = element.lower().strip()
                if element not in flattened_interactables_map:
                    flattened_interactables_map[stripped_lowered_element] = len(flattened_interactables) - 1
                elements.append(stripped_lowered_element)
        except Exception as e:
            pass

    for i, element in enumerate(elements):
        cleaned_elements.append(clean_element(element))
    if limit_elements:
        BM.mark('limit_elements')
        elements, limit_strings, input_tokens, output_tokens = actor.limit_elements(elements)
        BM.mark('limit_elements')
    print('LIMITED ELEMENTS: {}'.format(len(elements)))
    has_true_element = False
    true_tag_id = None
    if true_element == element or len(get_close_matches(true_element, elements, n=1, cutoff=0.95)) > 0:
        has_true_element = True
        true_tag_id = i
    for i, element in enumerate(elements):
        cleansed_element = clean_element(element)
        cleaned_elements_map[cleansed_element] = element
        encoded_element = ENCODING.encode(cleansed_element)
        if current_tokens + len(encoded_element) <= actor.MAX_TOKENS:
            current_sublist.append(cleansed_element)
            current_tokens += len(encoded_element)
        else:
            sublists.append(current_sublist)
            current_sublist = [cleansed_element]
            current_tokens = len(encoded_element)
    if current_sublist:
        sublists.append(current_sublist)

    BM.mark('filter_elements')
    filtered_html = ''
    with open('data/examples/all_elements.html', 'w') as outfile:
        counter = 1
        for element, _ in all_elements.items():
            filtered_html += '{}\n'.format(element)
            outfile.write('({}) {}\n\n'.format(counter, element))
            counter += 1
            
    cleaned_html = ''
    with open('data/examples/elements.html', 'w') as outfile:
        counter = 1
        for cleaned_element in cleaned_elements:
            cleaned_html += '{}\n'.format(cleaned_element)
            outfile.write('({}) {}\n\n'.format(counter, cleaned_element))
            counter += 1

    limited_html = ''
    with open('data/examples/limited_elements.html', 'w') as outfile:
        counter = 1
        for sublist in sublists:
            for i, element in enumerate(sublist):
                limited_html += '({}) {}\n\n'.format(i, element)
                outfile.write('({}) {}\n\n'.format(counter, element))
                counter += 1

    return sublists, filtered_html, cleaned_html, limited_html, input_tokens, output_tokens, has_true_element, true_tag_id, limit_strings


global OAI
global OAI_LONG
global OAI_GOOD
global ENCODING
global EXIT_BOOL
OAI = OpenAIAPI(model='gpt-3.5-turbo', mode='azure')
OAI_LONG = OpenAIAPI(model='gpt-3.5-turbo-16k', mode='azure')
OAI_GOOD = OpenAIAPI(model='gpt-4')
ENCODING = tiktoken.encoding_for_model('gpt-3.5-turbo-16k')
EXIT_BOOL = False
signal.signal(signal.SIGQUIT, interrupt_handler)


def gpt_score(true_output, pred_output):
    message, _ = OAI_GOOD.handle_response('Take a deep breath. If the following outputs of these two different language models are similar, respond with "Similar". Otherwise, reply with "Different".', 'First output: {}\Second output: {}'.format(true_output, pred_output))
    if message.startswith('Similar'):
        return True
    elif message.startswith('Different'):
        return False
    return message


def gpt_score_element(true_output, pred_output):
    message, _ = OAI_GOOD.handle_response('Take a deep breath. If the following two HTML elements are semantically similar (same label/text and/or same purpose), respond with "Similar". Otherwise, reply with "Different". E.g., "<label role="presentation">10 miles </label>" and "<li class="dropdownlist-item_iskeyboardfocused dropdownlist-item" id="10 miles" role="option">10 miles </li>" are similar. But "<label class="searchsuggestion" role="presentation">94587 , ca </label>" and "<input class="field-input inputfield" id="simplesearchlocation" name="simplesearchlocation" placeholder="enter city, state, or zip" role="combobox" type="text" value="">" are different.', 'First output: {}\Second output: {}'.format(true_output, pred_output))
    if message.startswith('Similar'):
        return True
    elif message.startswith('Different'):
        return False
    return message


ACTION_VERB_MAP = {'click': 'click', 'type': 'type_text', 'drag': 'drag', 'check': 'check_item', 'enter': 'press_enter', 'right_click': 'right_click', 'select': 'select_option', 'visit_url': 'visit_url', 'upload_file': 'upload_file', 'type_and_enter': 'type_and_enter', 'type_text': 'type_text', 'check_item': 'check_item', 'press_enter': 'press_enter', 'select_option': 'select_option'}

sample_list = json.load(open('manual_annotations.json', 'r'))

files = os.listdir('../alpaca_datasets/Mind2Web/data/test/')

candidate_proposals = []
candidate_proposals_long = []
candidate_action_selections = []
end_state_terminations = []
makes_senses = []
page_contexts = []
secondary_parameters = []
high_level_elements = []
limiter_strings = []
next_actions = []
tokens = {'process_screenshot': {'input': [], 'output': []}, 'page_context': {'input': [], 'output': []}, 'limit_elements': {'input': [], 'output': []}, 'end_state': {'input': [], 'output': []}, 'text_field': {'input': [], 'output': []}, 'select_option': {'input': [], 'output': []}, 'raw_html': [], 'mind2web_html': [], 'filtered_html': [], 'clean_html': [], 'limited_html': []}
elements_dict = {'is_in_interactables': [], 'is_in_limited_elements': []}
done_annotations = {}
skip_annotations = {}
scores = {'total_tasks': 0, 'total_actions': 0, 'element_proposal_match': 0, 'element_proposal_total': 0, 'action_element_selection_match': 0, 'action_element_selection_total': 0, 'action_selection_match': 0, 'element_selection_match': 0, 'text_field_match': 0, 'text_field_total': 0, 'select_option_match': 0, 'select_option_total': 0, 'makes_sense_match': 0, 'makes_sense_total': 0, 'end_state_tp': 0, 'end_state_fp': 0, 'end_state_tn': 0, 'end_state_fn': 0, 'correct_actions': 0, 'correct_tasks': 0, 'cur_correct_actions': 0, 'correct_proportion': 0.0, 'cur_total_actions': 0, 'element_proposal_match_close': 0, 'element_selection_match_close': 0, 'text_field_match_close': 0, 'select_option_match_close': 0, 'action_element_selection_match_close': 0, 'task_total_actions': 0, 'task_correct_actions': 0}
proportions = []
task_proportions = []

out_data = []
for file in files:
    if file.startswith('test_domain'):
        newfile = 'test_domain'
    elif file.startswith('test_task'):
        newfile = 'test_task'
    elif file.startswith('test_website'):
        newfile = 'test_website'
    else:
        continue

    with open('../alpaca_datasets/Mind2Web/data/test/{}'.format(file), 'r') as infile:
        data = json.load(infile)
        for item in data:
            if EXIT_BOOL:
                exit(0)
            if item['annotation_id'] not in sample_list:
                continue
            if item['annotation_id'] not in done_annotations:
                done_annotations[item['annotation_id']] = []
            if not item['website'].endswith('.org') and not item['website'].endswith('.net') and not item['website'].endswith('.info') and not item['website'].endswith('.edu') and not item['website'].endswith('.gov') and not item['website'].endswith('.io') and not item['website'].endswith('.name') and not item['website'].endswith('.biz') and not item['website'].endswith('.mil') and not item['website'].endswith('.int') and not item['website'].endswith('.us') and not item['website'].endswith('.ca') and not item['website'].endswith('.uk') and not item['website'].endswith('.eu') and not item['website'].endswith('.au') and not item['website'].endswith('.cn') and not item['website'].endswith('.in') and not item['website'].endswith('.br') and not item['website'].endswith('.ru') and not item['website'].endswith('.jp') and not item['website'].endswith('.website') and not item['website'].endswith('.company') and not item['website'].endswith('.global') and not item['website'].endswith('.online') and not item['website'].endswith('.site') and not item['website'].endswith('.tech') and not item['website'].endswith('.news') and not item['website'].endswith('.blog') and not item['website'].endswith('.app') and not item['website'].endswith('.pro') and not item['website'].endswith('.museum') and not item['website'].endswith('.coop') and not item['website'].endswith('.aero') and not item['website'].endswith('.jobs') and not item['website'].endswith('.mobi') and not item['website'].endswith('.store') and not item['website'].endswith('.music') and not item['website'].endswith('.movie') and not item['website'].endswith('.game') and not item['website'].endswith('.fun') and not item['website'].endswith('.ai') and not item['website'].endswith('.books') and not item['website'].endswith('.sport') and not item['website'].endswith('.school') and not item['website'].endswith('.cloud') and not item['website'].endswith('.host') and not item['website'].endswith('.dev') and not item['website'].endswith('.design') and not item['website'].endswith('.photography') and not item['website'].endswith('.realty') and not item['website'].endswith('.travel') and not item['website'].endswith('.health') and not item['website'].endswith('.de') and not item['website'].endswith('.fr') and not item['website'].endswith('.it') and not item['website'].endswith('.es') and not item['website'].endswith('.nl') and not item['website'].endswith('.se') and not item['website'].endswith('.no') and not item['website'].endswith('.il') and not item['website'].endswith('.sg') and not item['website'].endswith('.za') and not item['website'].endswith('.com'):
                website = item['website'] + '.com'
            else:
                website = item['website']
            task = item['confirmed_task']
            out_data.append({'website': website, 'task': task})
            continue
with open('data/manual_tasks.json', 'w') as outfile:
    json.dump(out_data, outfile, indent=4)
exit()