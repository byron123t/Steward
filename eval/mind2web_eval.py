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

import json, os, re, argparse, tiktoken, csv
import pandas as pd
from tqdm import tqdm
import random
from data import prompts
from src.html_processing import clean_element
from src.processor.html.parse_html import ParseHtml
from src.smart_runtime import PageStateActor
from src.data_collector.API import OpenAIAPI
from sensitive.objects import *


random.seed(42)

global OAI
global OAI_LONG
global OAI_GOOD
global OAI_MAKES_SENSE
global OAI_END_STATE
global ENCODING
OAI = OpenAIAPI(model='gpt-3.5-turbo', mode='azure')
OAI_LONG = OpenAIAPI(model='gpt-3.5-turbo-16k', mode='azure')
OAI_GOOD = OpenAIAPI(model='gpt-4')
OAI_MAKES_SENSE = OpenAIAPI(model='ft:gpt-3.5-turbo-0613:university-of-michigan:small-make-sense:83AOHkG4')
OAI_END_STATE = OpenAIAPI(model='ft:gpt-3.5-turbo-0613:university-of-michigan:small-end-state:838yV4xV')
ENCODING = tiktoken.encoding_for_model('gpt-3.5-turbo-16k')

ACTION_VERB_MAP = {'click': 'click', 'type': 'type_text', 'drag': 'drag', 'check': 'check_item', 'enter': 'press_enter', 'right_click': 'right_click', 'select': 'select_option', 'visit_url': 'visit_url', 'upload_file': 'upload_file', 'type_and_enter': 'type_and_enter', 'type_text': 'type_text', 'check_item': 'check_item', 'press_enter': 'press_enter', 'select_option': 'select_option', 'hover': 'hover'}
global_actions = ["type_and_enter", "click", "type_text", "drag", "check_item", "press_enter", "right_click", "select_option", "visit_url", "upload_file", "copy", "paste"]


def shuffle_list_by_indices(lst):
    shuffled_indices = list(range(len(lst)))
    random.shuffle(shuffled_indices)
    shuffled_list = [lst[i] for i in shuffled_indices]
    return shuffled_indices, shuffled_list


def append_to_pandas(pandas_dict, kwargs):
    for key in kwargs:
        pandas_dict[key].append(kwargs[key])
    return pandas_dict


def save_each_to_csv(data, filename):
    with open(filename, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['instruction', 'input', 'pred_output', 'true_output', 'correct'])
        writer.writeheader()
        for item in data:
            writer.writerow(item)


def save_to_csv(data, filename):
    with open(filename, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['tp', 'tn', 'fp', 'fn', 'action_correct', 'element_correct', 'correct', 'none_correct', 'total', 'none_total', 'perplexity', 'none_accuracy', 'element_accuracy', 'action_accuracy', 'accuracy', 'tokens', 'episode'])
        writer.writeheader()
        for item in data:
            writer.writerow(item)


def gpt_score(true_output, pred_output):
    message, _ = OAI.handle_response('Take a deep breath. If the following outputs of these two different language models are similar, respond with "Similar". Otherwise, reply with "Different".', 'First output: {}\Second output: {}'.format(true_output, pred_output))
    if message.startswith('Similar'):
        return True
    elif message.startswith('Different'):
        return False
    return message


def gpt_score_element(true_output, pred_output):
    message, _ = OAI.handle_response('Take a deep breath. If the following two HTML elements are semantically similar (same inner HTML text and/or same purpose), respond with "Similar". Otherwise, reply with "Different".', 'First output: {}\Second output: {}'.format(true_output, pred_output))
    if message.startswith('Similar'):
        return True
    elif message.startswith('Different'):
        return False
    return message


scores = {'total_tasks': 0, 'total_actions': 0, 'element_proposal_match': 0, 'element_proposal_total': 0, 'action_element_selection_match': 0, 'action_element_selection_total': 0, 'action_selection_match': 0, 'element_selection_match': 0, 'text_field_match': 0, 'text_field_total': 0, 'select_option_match': 0, 'select_option_total': 0, 'makes_sense_match': 0, 'makes_sense_total': 0, 'end_state_tp': 0, 'end_state_fp': 0, 'end_state_tn': 0, 'end_state_fn': 0, 'correct_actions': 0, 'correct_tasks': 0, 'cur_correct_actions': 0, 'correct_proportion': 0.0, 'cur_total_actions': 0, 'element_proposal_match_close': 0, 'element_selection_match_close': 0, 'text_field_match_close': 0, 'select_option_match_close': 0, 'action_element_selection_match_close': 0}
proportions = []
task_proportions = []
pandas_scores = {'website': [], 'domain': [], 'subdomain': [], 'task':[], 'task_id': [], 'action_id': [], 'action_repr': [], 'action_step': [], 'component': [], 'correct': [], 'close_correct': [], 'cur_total_actions': [], 'component_accuracy': [], 'close_component_accuracy': [], 'action_accuracy': [], 'element_accuracy': [], 'close_element_accuracy': [], 'tokens': [], 'true_output_idx': [], 'pred_output_idx': [], 'true_output': [], 'pred_output': []}
in_interactables = []
all_data = []

if os.path.exists('data/results/mind2web/e2e_alldata.csv'):
    with open('data/results/mind2web/e2e_alldata.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            all_data.append(row)
            for key, val in row.items():
                if len(key) > 0:
                    pandas_scores[key].append(val)
if os.path.exists('data/results/mind2web/e2escores.json'):
    with open('data/results/mind2web/e2escores.json', 'r') as infile:
        scores = json.load(infile)
if os.path.exists('data/results/mind2web/e2eproportions.csv'):
    with open('data/results/mind2web/e2eactionproportions.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            proportions.append(row)
if os.path.exists('data/results/mind2web/e2etaskproportions.csv'):
    with open('data/results/mind2web/e2etaskproportions.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            task_proportions.append(row)
if os.path.exists('data/results/mind2web/in_interactables.json'):
    with open('data/results/mind2web/in_interactables.json', 'r') as infile:
        in_interactables = json.load(infile)

for file in tqdm(os.listdir('../alpaca_datasets/Mind2Web/data/test/')):
    if file.endswith('.json'):
        with open('../alpaca_datasets/Mind2Web/data/test/{}'.format(file), 'r') as infile:
            data = json.load(infile)
            random.shuffle(data)
            for item in tqdm(data[:5]):
                if not item['website'].endswith('.org') and not item['website'].endswith('.net') and not item['website'].endswith('.info') and not item['website'].endswith('.edu') and not item['website'].endswith('.gov') and not item['website'].endswith('.io') and not item['website'].endswith('.name') and not item['website'].endswith('.biz') and not item['website'].endswith('.mil') and not item['website'].endswith('.int') and not item['website'].endswith('.us') and not item['website'].endswith('.ca') and not item['website'].endswith('.uk') and not item['website'].endswith('.eu') and not item['website'].endswith('.au') and not item['website'].endswith('.cn') and not item['website'].endswith('.in') and not item['website'].endswith('.br') and not item['website'].endswith('.ru') and not item['website'].endswith('.jp') and not item['website'].endswith('.website') and not item['website'].endswith('.company') and not item['website'].endswith('.global') and not item['website'].endswith('.online') and not item['website'].endswith('.site') and not item['website'].endswith('.tech') and not item['website'].endswith('.news') and not item['website'].endswith('.blog') and not item['website'].endswith('.app') and not item['website'].endswith('.pro') and not item['website'].endswith('.museum') and not item['website'].endswith('.coop') and not item['website'].endswith('.aero') and not item['website'].endswith('.jobs') and not item['website'].endswith('.mobi') and not item['website'].endswith('.store') and not item['website'].endswith('.music') and not item['website'].endswith('.movie') and not item['website'].endswith('.game') and not item['website'].endswith('.fun') and not item['website'].endswith('.ai') and not item['website'].endswith('.books') and not item['website'].endswith('.sport') and not item['website'].endswith('.school') and not item['website'].endswith('.cloud') and not item['website'].endswith('.host') and not item['website'].endswith('.dev') and not item['website'].endswith('.design') and not item['website'].endswith('.photography') and not item['website'].endswith('.realty') and not item['website'].endswith('.travel') and not item['website'].endswith('.health') and not item['website'].endswith('.de') and not item['website'].endswith('.fr') and not item['website'].endswith('.it') and not item['website'].endswith('.es') and not item['website'].endswith('.nl') and not item['website'].endswith('.se') and not item['website'].endswith('.no') and not item['website'].endswith('.il') and not item['website'].endswith('.sg') and not item['website'].endswith('.za') and not item['website'].endswith('.com'):
                    website = item['website'] + '.com'
                else:
                    website = item['website']
                task = item['confirmed_task']
                print(task)
                action_representations = item['action_reprs']
                print(action_representations)
                parser = ParseHtml(item['actions'][0]['raw_html'])
                interactables = parser.get_interactables()
                actor = PageStateActor(website, task, interactables, item['actions'][0]['raw_html'], 12000, False)
                scores['total_tasks'] += 1
                scores['cur_correct_actions'] = 0
                scores['cur_total_actions'] = len(item['actions'])
                
                progress_bar = tqdm(enumerate(item['actions']), total=len(item['actions']))
                for action_id, action in progress_bar:
                    skip = False
                    for item_data in all_data:
                        if action['action_uid'] in item_data['action_id']:
                            skip = True

                    selected_action = ACTION_VERB_MAP[action['operation']['op'].lower()]
                    if selected_action not in ['click', 'type_text', 'select_option']:
                        scores['cur_total_actions'] -= 1
                        continue
                    secondary_parameter = action['operation']['value']
                    node_id = None
                    for candidate in action['pos_candidates']:
                        node_id = candidate['backend_node_id']
                    with open('data/examples/raw.html', 'w') as outfile:
                        outfile.write(action['raw_html'])
                    with open('data/examples/cleaned.html', 'w') as outfile:
                        outfile.write(action['cleaned_html'])
                    parser = ParseHtml(action['raw_html'])
                    if not node_id:
                        scores['cur_total_actions'] -= 1
                        continue
                    
                    scores['total_actions'] += 1
                    selected_element = parser.soup.select_one('[backend_node_id="{}"]'.format(node_id))
                    backend_node_id = selected_element['backend_node_id']
                    if selected_action.startswith('select'):
                        try:
                            options = parser.soup.select_one('[backend_node_id="{}"]'.format(node_id)).find_all('option')
                        except Exception as e:
                            options = None
                    del selected_element.attrs['backend_node_id']
                    selected_element_str = clean_element(selected_element.prettify())
                    selected_element.attrs['backend_node_id'] = backend_node_id

                    if skip:
                        true_high_level_element, _ = actor.get_high_level_element(selected_element_str, selected_action)
                        true_high_level_action = '{} {}'.format(selected_action, true_high_level_element)
                        if selected_action.startswith('type') or selected_action.startswith('select'):
                            true_high_level_action = '{} {} in {}'.format(selected_action, secondary_parameter, selected_element_str)
                        actor.ACTIONS.append(true_high_level_action)
                        continue

                    interactables = parser.get_interactables(clean=False)
                    is_in_interactables = False
                    true_element_tag_id = None
                    for i, element in enumerate(interactables):
                        if 'backend_node_id' in element.attrs:
                            if element.attrs['backend_node_id'] == node_id:
                                is_in_interactables = True
                                true_element_tag_id = i
                            backend_node_id = element.attrs['backend_node_id']
                            del element.attrs['backend_node_id']
                    if not is_in_interactables:
                        interactables.append(selected_element)
                        true_element_tag_id = len(interactables) - 1
                        print(selected_element_str)
                        in_interactables.append({'element': selected_element_str, 'is_in_interactables': False})
                    else:
                        in_interactables.append({'element': selected_element_str, 'is_in_interactables': True})

                    actor.update(interactables=interactables, html=action['raw_html'])

                    sublists = []
                    current_sublist = []
                    current_tokens = 0
                    for element in interactables:
                        to_append = True
                        cleansed_element = clean_element(element.prettify())
                        encoded_element = ENCODING.encode(cleansed_element)
                        if current_tokens + len(encoded_element) <= 10000:
                            current_sublist.append(cleansed_element)
                            current_tokens += len(encoded_element)
                        else:
                            to_append = False
                            sublists.append(current_sublist)
                            current_sublist = [cleansed_element]
                            current_tokens = len(encoded_element)
                    if to_append:
                        sublists.append(current_sublist)

                    correct_dict = {'element_proposal': False, 'action_element_selection': False, 'makes_sense': False, 'end_state': False, 'secondary_parameter': False}
                    tag_id_list, element_list, token_count = actor.element_proposal(sublists, n_tries=10)
                    if int(true_element_tag_id) in tag_id_list:
                        scores['element_proposal_match'] += 1
                        correct_dict['element_proposal'] = True
                    else:
                        print(true_element_tag_id, tag_id_list)
                        # continue
                    scores['element_proposal_total'] += 1
                    pandas_scores = append_to_pandas(pandas_scores, {'website': website, 'domain': item['domain'], 'subdomain': item['subdomain'], 'task': task, 'action_repr': action_representations, 'task_id': item['annotation_id'], 'action_id': action['action_uid'], 'action_step': action_id, 'component': 'element_proposal', 'correct': correct_dict['element_proposal'], 'close_correct': None, 'cur_total_actions': scores['cur_total_actions'], 'component_accuracy': scores['element_proposal_match'] / scores['element_proposal_total'], 'close_component_accuracy': None, 'tokens': token_count, 'true_output_idx': true_element_tag_id, 'pred_output_idx': tag_id_list, 'true_output': None, 'pred_output': None, 'action_accuracy': None, 'element_accuracy': None, 'close_element_accuracy': None})
                    
                    tag_id, newelement, _, verb, message, token_count = actor.action_element_selection(tag_id_list, element_list)
                    print(selected_action, verb, selected_element_str, newelement)

                    if tag_id is None:
                        tag_id, newelement, _, verb, message, token_count = actor.action_element_selection(tag_id_list, element_list)
                        if tag_id is None:
                            tag_id, newelement, _, verb, message, token_count = actor.action_element_selection(tag_id_list, element_list)
                            if tag_id is None:
                                continue

                    action_correct = False
                    close_correct = False
                    if verb == selected_action and tag_id == true_element_tag_id:
                        correct_dict['action_element_selection'] = True
                        scores['action_element_selection_match'] += 1
                        scores['action_element_selection_match_close'] += 1
                        print(selected_action, verb, selected_element_str, newelement)
                    if verb == selected_action:
                        scores['action_selection_match'] += 1
                        action_correct = True
                    if tag_id == true_element_tag_id:
                        scores['element_selection_match'] += 1
                        scores['element_selection_match_close'] += 1
                    else:
                        if gpt_score_element(selected_element_str, newelement):
                            if action_correct:
                                scores['action_element_selection_match_close'] += 1
                                close_correct = True
                            scores['element_selection_match_close'] += 1
                    scores['action_element_selection_total'] += 1
                    newtag = interactables[tag_id]
                    pandas_scores = append_to_pandas(pandas_scores, {'website': website, 'domain': item['domain'], 'subdomain': item['subdomain'], 'task': task, 'action_repr': action_representations, 'task_id': item['annotation_id'], 'action_id': action['action_uid'], 'action_step': action_id, 'component': 'action_element_selection', 'correct': correct_dict['action_element_selection'], 'close_correct': close_correct, 'cur_total_actions': scores['cur_total_actions'], 'component_accuracy': scores['action_element_selection_match'] / scores['action_element_selection_total'], 'close_component_accuracy': scores['action_element_selection_match_close'] / scores['action_element_selection_total'], 'tokens': token_count, 'true_output_idx': true_element_tag_id, 'pred_output_idx': tag_id, 'true_output': selected_element_str, 'pred_output': newelement, 'action_accuracy': scores['action_selection_match'] / scores['action_element_selection_total'], 'element_accuracy': scores['element_proposal_match'] / scores['action_element_selection_total'], 'close_element_accuracy': scores['action_element_selection_match_close'] / scores['action_element_selection_total']})

                    candidate = '{} {}'.format(verb, newelement)
                    high_level_element, token_count = actor.get_high_level_element(candidate, verb)
                    high_level_action = '{} {}'.format(verb, high_level_element)
                    
                    print(high_level_action)
                    if actor.verbose: print(candidate)
                    if actor.verbose: print('HIGH LEVEL ACTION\n{}\n\n'.format(high_level_action))
                    pandas_scores = append_to_pandas(pandas_scores, {'website': website, 'domain': item['domain'], 'subdomain': item['subdomain'], 'task': task, 'action_repr': action_representations, 'task_id': item['annotation_id'], 'action_id': action['action_uid'], 'action_step': action_id, 'component': 'high_level', 'correct': None, 'close_correct': None, 'cur_total_actions': scores['cur_total_actions'], 'component_accuracy': None, 'close_component_accuracy': None, 'tokens': token_count, 'true_output_idx': None, 'pred_output_idx': None, 'true_output': None, 'pred_output': None, 'action_accuracy': None, 'element_accuracy': None, 'close_element_accuracy': None})

                    text_field_censored = None
                    text_field = None
                    if verb.startswith('type'):
                        text_field_message, text_field, text_field_censored, token_count = actor.text_fields(high_level_action, verb)
                        if text_field_censored:
                            high_level_action = '{} {} in {}'.format(verb, text_field_censored, high_level_element)
                        if actor.verbose: print('TEXT FIELD INPUT\n{}\n\n'.format(text_field_censored))
                        if actor.verbose: print('========================================\n')
                        if selected_action.startswith('type'):
                            if text_field_censored:
                                if secondary_parameter.lower().strip() in text_field_censored.lower().strip() or text_field_censored.lower().strip() in secondary_parameter.lower().strip():
                                    scores['text_field_match'] += 1
                                    correct_dict['secondary_parameter'] = True
                                else:
                                    result = gpt_score(secondary_parameter, text_field_censored)
                                    if result:
                                        correct_dict['secondary_parameter'] = True
                                        scores['text_field_match'] += 1
                                    else:
                                        print(secondary_parameter, text_field_censored)
                            scores['text_field_total'] += 1
                            pandas_scores = append_to_pandas(pandas_scores, {'website': website, 'domain': item['domain'], 'subdomain': item['subdomain'], 'task': task, 'action_repr': action_representations, 'task_id': item['annotation_id'], 'action_id': action['action_uid'], 'action_step': action_id, 'component': 'text_field', 'correct': correct_dict['secondary_parameter'], 'close_correct': None, 'cur_total_actions': scores['cur_total_actions'], 'component_accuracy': scores['text_field_match'] / scores['text_field_total'], 'close_component_accuracy': None, 'tokens': token_count, 'true_output_idx': None, 'pred_output_idx': None, 'true_output': secondary_parameter, 'pred_output': text_field, 'action_accuracy': None, 'element_accuracy': None, 'close_element_accuracy': None})
                    elif verb.startswith('select'):
                        if selected_action.startswith('select_option'):
                            if options:
                                select_option_message, select_option, token_count = actor.select_options(high_level_action, verb, options)
                                if select_option:
                                    high_level_action = '{} {} in {}'.format(verb, select_option, high_level_element)
                                if select_option == secondary_parameter:
                                    correct_dict['secondary_parameter'] = True
                                    scores['select_option_match'] += 1
                                else:
                                    print(secondary_parameter, select_option)
                            scores['select_option_total'] += 1
                            pandas_scores = append_to_pandas(pandas_scores, {'website': website, 'domain': item['domain'], 'subdomain': item['subdomain'], 'task': task, 'action_repr': action_representations, 'task_id': item['annotation_id'], 'action_id': action['action_uid'], 'action_step': action_id, 'component': 'select_option', 'correct': correct_dict['secondary_parameter'], 'close_correct': None, 'cur_total_actions': scores['cur_total_actions'], 'component_accuracy': scores['select_option_match'] / scores['select_option_total'], 'close_component_accuracy': None, 'tokens': token_count, 'true_output_idx': None, 'pred_output_idx': None, 'true_output': secondary_parameter, 'pred_output': select_option, 'action_accuracy': None, 'element_accuracy': None, 'close_element_accuracy': None})
                    else:
                        correct_dict['secondary_parameter'] = True
                        
                    if correct_dict['action_element_selection'] and correct_dict['secondary_parameter']:
                        scores['correct_actions'] += 1
                        scores['cur_correct_actions'] += 1
                        scores['correct_proportion'] = scores['correct_actions'] / scores['total_actions']

                    scores['makes_sense_total'] += 1
                    correct_dict['makes_sense'] = False
                    makes_sense, makes_sense_message, token_count = actor.makes_sense(verb, newelement, text_field_censored)
                    
                    true_high_level_element, _ = actor.get_high_level_element(selected_element_str, selected_action)
                    true_high_level_action = '{} {}'.format(selected_action, true_high_level_element)
                    if selected_action.startswith('type') or selected_action.startswith('select'):
                        true_high_level_action = '{} {} in {}'.format(selected_action, secondary_parameter, selected_element_str)
                    actor.ACTIONS.append(true_high_level_action)
                    if makes_sense:
                        correct_dict['makes_sense'] = True
                        scores['makes_sense_match'] += 1
                        actor.newtag = newtag
                        actor.newelement = newelement
                        actor.action = action
                        actor.text_field = text_field
                        pandas_scores = append_to_pandas(pandas_scores, {'website': website, 'domain': item['domain'], 'subdomain': item['subdomain'], 'task': task, 'action_repr': action_representations, 'task_id': item['annotation_id'], 'action_id': action['action_uid'], 'action_step': action_id, 'component': 'makes_sense', 'correct': correct_dict['makes_sense'], 'close_correct': None, 'cur_total_actions': scores['cur_total_actions'], 'component_accuracy': scores['makes_sense_match'] / scores['makes_sense_total'], 'close_component_accuracy': None, 'tokens': token_count, 'true_output_idx': True, 'pred_output_idx': True, 'true_output': 'Yes', 'pred_output': makes_sense_message, 'action_accuracy': None, 'element_accuracy': None, 'close_element_accuracy': None})
                        
                        end_state, end_state_message, token_count = actor.end_state()
                        if end_state:
                            if len(actor.ACTIONS) == len(item['actions']) - 1:
                                true_output = 'Yes'
                                true_output_idx = True
                                correct_dict['end_state'] = True
                                scores['end_state_tp'] += 1
                            else:
                                true_output = 'No'
                                true_output_idx = False
                                scores['end_state_fp'] += 1
                        else:
                            if len(actor.ACTIONS) < len(item['actions']) - 1:
                                true_output = 'No'
                                true_output_idx = False
                                correct_dict['end_state'] = True
                                scores['end_state_tn'] += 1
                            else:
                                true_output = 'Yes'
                                true_output_idx = True
                                scores['end_state_fn'] += 1
                        end_state_acc = (scores['end_state_tp'] + scores['end_state_tn']) / (scores['end_state_tp'] + scores['end_state_tn'] + scores['end_state_fp'] + scores['end_state_fn'])
                        pandas_scores = append_to_pandas(pandas_scores, {'website': website, 'domain': item['domain'], 'subdomain': item['subdomain'], 'task': task, 'action_repr': action_representations, 'task_id': item['annotation_id'], 'action_id': action['action_uid'], 'action_step': action_id, 'component': 'end_state', 'correct': correct_dict['end_state'], 'close_correct': None, 'cur_total_actions': scores['cur_total_actions'], 'component_accuracy': end_state_acc, 'close_component_accuracy': None, 'tokens': token_count, 'true_output_idx': true_output_idx, 'pred_output_idx': end_state, 'true_output': true_output, 'pred_output': end_state_message, 'action_accuracy': None, 'element_accuracy': None, 'close_element_accuracy': None})
                    else:
                        pandas_scores = append_to_pandas(pandas_scores, {'website': website, 'domain': item['domain'], 'subdomain': item['subdomain'], 'task': task, 'action_repr': action_representations, 'task_id': item['annotation_id'], 'action_id': action['action_uid'], 'action_step': action_id, 'component': 'makes_sense', 'correct': correct_dict['makes_sense'], 'close_correct': None, 'cur_total_actions': scores['cur_total_actions'], 'component_accuracy': scores['makes_sense_match'] / scores['makes_sense_total'], 'close_component_accuracy': None, 'tokens': token_count, 'true_output_idx': True, 'pred_output_idx': False, 'true_output': 'Yes', 'pred_output': makes_sense_message, 'action_accuracy': None, 'element_accuracy': None, 'close_element_accuracy': None})
                        
                    pd.DataFrame(pandas_scores).to_csv('data/results/mind2web/e2e_alldata.csv')
                    with open('data/results/mind2web/in_interactables.json', 'w') as outfile:
                        json.dump(in_interactables, outfile, indent=4)
                    with open('data/results/mind2web/e2escores.json', 'w') as outfile:
                        json.dump(scores, outfile, indent=4)
                    proportions.append(scores['correct_proportion'])
                    with open('data/results/mind2web/e2eactionproportions.csv', 'w') as outfile:
                        csv.DictWriter(outfile, fieldnames=['correct_proportion']).writeheader()
                        for proportion in proportions:
                            csv.DictWriter(outfile, fieldnames=['correct_proportion']).writerow({'correct_proportion': proportion})
                    if makes_sense:
                        continue
                if scores['correct_proportion'] == 1.0:
                    scores['correct_tasks'] += 1
                task_proportions.append(scores['correct_tasks'] / scores['total_tasks'])
                with open('data/results/mind2web/e2etaskproportions.csv', 'w') as outfile:
                    csv.DictWriter(outfile, fieldnames=['correct_proportion']).writeheader()
                    for proportion in task_proportions:
                        csv.DictWriter(outfile, fieldnames=['correct_proportion']).writerow({'correct_proportion': proportion})
