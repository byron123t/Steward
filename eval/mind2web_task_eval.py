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


import json, os, re, signal, argparse, tiktoken
import pandas as pd
from tqdm import tqdm
from random import random, shuffle, randrange, choice
from data import prompts
from src.html_processing import clean_element
from src.processor.html.parse_html import ParseHtml
from src.smart_runtime import PageStateActor
from src.data_collector.API import OpenAIAPI
from sensitive.objects import *


def interrupt_handler(signum, frame):
    global EXIT_BOOL
    print("\nInterrupt signal received. Finishing up and terminating...")
    EXIT_BOOL = True


def shuffle_list_by_indices(lst):
    shuffled_indices = list(range(len(lst)))
    shuffle(shuffled_indices)
    shuffled_list = [lst[i] for i in shuffled_indices]
    return shuffled_indices, shuffled_list


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


ACTION_VERB_MAP = {'click': 'click', 'type': 'type_text', 'drag': 'drag', 'check': 'check_item', 'enter': 'press_enter', 'right_click': 'right_click', 'select': 'select_option', 'visit_url': 'visit_url', 'upload_file': 'upload_file', 'type_and_enter': 'type_and_enter', 'type_text': 'type_text', 'check_item': 'check_item', 'press_enter': 'press_enter', 'select_option': 'select_option'}

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=int, default=0)
args = parser.parse_args()

file = os.listdir('../alpaca_datasets/Mind2Web/data/train/')[args.file]
if not os.path.exists('../alpaca_datasets/alpaca2web/{}/'.format(file.replace('.json', ''))):
    os.mkdir('../alpaca_datasets/alpaca2web/{}/'.format(file.replace('.json', '')))
    candidate_proposals = []
    candidate_proposals_long = []
    candidate_action_selections = []
    end_state_terminations = []
    makes_senses = []
    page_contexts = []
    secondary_parameters = []
    high_level_elements = []
    done_annotations = {}
    skip_annotations = {}
    with open('../alpaca_datasets/alpaca2web/{}/candidate_proposal.json'.format(file.replace('.json', '')), 'w') as outfile:
        json.dump(candidate_proposals, outfile, indent=4)
    with open('../alpaca_datasets/alpaca2web/{}/candidate_proposal_long.json'.format(file.replace('.json', '')), 'w') as outfile:
        json.dump(candidate_proposals_long, outfile, indent=4)
    with open('../alpaca_datasets/alpaca2web/{}/candidate_action_selection.json'.format(file.replace('.json', '')), 'w') as outfile:
        json.dump(candidate_action_selections, outfile, indent=4)
    with open('../alpaca_datasets/alpaca2web/{}/end_state_termination.json'.format(file.replace('.json', '')), 'w') as outfile:
        json.dump(end_state_terminations, outfile, indent=4)
    with open('../alpaca_datasets/alpaca2web/{}/makes_sense.json'.format(file.replace('.json', '')), 'w') as outfile:
        json.dump(makes_senses, outfile, indent=4)
    with open('../alpaca_datasets/alpaca2web/{}/page_context.json'.format(file.replace('.json', '')), 'w') as outfile:
        json.dump(page_contexts, outfile, indent=4)
    with open('../alpaca_datasets/alpaca2web/{}/secondary_parameter.json'.format(file.replace('.json', '')), 'w') as outfile:
        json.dump(secondary_parameters, outfile, indent=4)
    with open('../alpaca_datasets/alpaca2web/{}/high_level_element.json'.format(file.replace('.json', '')), 'w') as outfile:
        json.dump(high_level_elements, outfile, indent=4)
    with open('../alpaca_datasets/alpaca2web/{}/done_annotations.json'.format(file.replace('.json', '')), 'w') as outfile:
        json.dump(done_annotations, outfile, indent=4)
    with open('../alpaca_datasets/alpaca2web/{}/skip_annotations.json'.format(file.replace('.json', '')), 'w') as outfile:
        json.dump(skip_annotations, outfile, indent=4)
else:
    with open('../alpaca_datasets/alpaca2web/{}/candidate_proposal.json'.format(file.replace('.json', '')), 'r') as infile:
        candidate_proposals = json.load(infile)
    with open('../alpaca_datasets/alpaca2web/{}/candidate_proposal_long.json'.format(file.replace('.json', '')), 'r') as infile:
        candidate_proposals_long = json.load(infile)
    with open('../alpaca_datasets/alpaca2web/{}/candidate_action_selection.json'.format(file.replace('.json', '')), 'r') as infile:
        candidate_action_selections = json.load(infile)
    with open('../alpaca_datasets/alpaca2web/{}/end_state_termination.json'.format(file.replace('.json', '')), 'r') as infile:
        end_state_terminations = json.load(infile)
    with open('../alpaca_datasets/alpaca2web/{}/makes_sense.json'.format(file.replace('.json', '')), 'r') as infile:
        makes_senses = json.load(infile)
    with open('../alpaca_datasets/alpaca2web/{}/page_context.json'.format(file.replace('.json', '')), 'r') as infile:
        page_contexts = json.load(infile)
    with open('../alpaca_datasets/alpaca2web/{}/secondary_parameter.json'.format(file.replace('.json', '')), 'r') as infile:
        secondary_parameters = json.load(infile)
    with open('../alpaca_datasets/alpaca2web/{}/high_level_element.json'.format(file.replace('.json', '')), 'r') as infile:
        high_level_elements = json.load(infile)
    with open('../alpaca_datasets/alpaca2web/{}/done_annotations.json'.format(file.replace('.json', '')), 'r') as infile:
        done_annotations = json.load(infile)
    with open('../alpaca_datasets/alpaca2web/{}/skip_annotations.json'.format(file.replace('.json', '')), 'r') as infile:
        skip_annotations = json.load(infile)

with open('../alpaca_datasets/Mind2Web/data/train/{}'.format(file), 'r') as infile:
    data = json.load(infile)
    for item in tqdm(data):
        if EXIT_BOOL:
            exit(0)
        if item['annotation_id'] not in done_annotations:
            done_annotations[item['annotation_id']] = []
        if not item['website'].endswith('.org') and not item['website'].endswith('.net') and not item['website'].endswith('.info') and not item['website'].endswith('.edu') and not item['website'].endswith('.gov') and not item['website'].endswith('.io') and not item['website'].endswith('.name') and not item['website'].endswith('.biz') and not item['website'].endswith('.mil') and not item['website'].endswith('.int') and not item['website'].endswith('.us') and not item['website'].endswith('.ca') and not item['website'].endswith('.uk') and not item['website'].endswith('.eu') and not item['website'].endswith('.au') and not item['website'].endswith('.cn') and not item['website'].endswith('.in') and not item['website'].endswith('.br') and not item['website'].endswith('.ru') and not item['website'].endswith('.jp') and not item['website'].endswith('.website') and not item['website'].endswith('.company') and not item['website'].endswith('.global') and not item['website'].endswith('.online') and not item['website'].endswith('.site') and not item['website'].endswith('.tech') and not item['website'].endswith('.news') and not item['website'].endswith('.blog') and not item['website'].endswith('.app') and not item['website'].endswith('.pro') and not item['website'].endswith('.museum') and not item['website'].endswith('.coop') and not item['website'].endswith('.aero') and not item['website'].endswith('.jobs') and not item['website'].endswith('.mobi') and not item['website'].endswith('.store') and not item['website'].endswith('.music') and not item['website'].endswith('.movie') and not item['website'].endswith('.game') and not item['website'].endswith('.fun') and not item['website'].endswith('.ai') and not item['website'].endswith('.books') and not item['website'].endswith('.sport') and not item['website'].endswith('.school') and not item['website'].endswith('.cloud') and not item['website'].endswith('.host') and not item['website'].endswith('.dev') and not item['website'].endswith('.design') and not item['website'].endswith('.photography') and not item['website'].endswith('.realty') and not item['website'].endswith('.travel') and not item['website'].endswith('.health') and not item['website'].endswith('.de') and not item['website'].endswith('.fr') and not item['website'].endswith('.it') and not item['website'].endswith('.es') and not item['website'].endswith('.nl') and not item['website'].endswith('.se') and not item['website'].endswith('.no') and not item['website'].endswith('.il') and not item['website'].endswith('.sg') and not item['website'].endswith('.za') and not item['website'].endswith('.com'):
            website = item['website'] + '.com'
        else:
            website = item['website']
        task = item['confirmed_task']
        # print(task)
        action_representations = item['action_reprs']
        # print(action_representations)
        parser = ParseHtml(item['actions'][0]['raw_html'])
        interactables = parser.get_interactables()
        actor = PageStateActor(website, task, interactables, item['actions'][0]['raw_html'], 12000, False)
        
        progress_bar = tqdm(enumerate(item['actions']), total=len(item['actions']))
        try:
            for action_id, action in progress_bar:
                if item['annotation_id'] in done_annotations:
                    if action['action_uid'] in done_annotations[item['annotation_id']]:
                        continue
                selected_action = ACTION_VERB_MAP[action['operation']['op'].lower()]
                secondary_parameter = action['operation']['value']
                node_id = None
                for candidate in action['pos_candidates']:
                    node_id = candidate['backend_node_id']
                with open('raw.html', 'w') as outfile:
                    outfile.write(action['raw_html'])
                with open('cleaned.html', 'w') as outfile:
                    outfile.write(action['cleaned_html'])
                parser = ParseHtml(action['raw_html'])
                if not node_id:
                    continue
                selected_element = parser.soup.select_one('[backend_node_id="{}"]'.format(node_id))
                interactables = parser.get_interactables()
                interactable_cleansed_elements = []
                interactable_map = {}
                current_sublist = []
                neg_current_sublist = []
                long_current_sublist = []
                neg_long_current_sublist = []
                current_tokens = 0
                long_current_tokens = 0
                shuffle(interactables)
                for element in interactables:
                    backend_node_id = element.attrs['backend_node_id']
                    del element.attrs['backend_node_id']
                    cleansed_element = clean_element(element.prettify())
                    element['backend_node_id'] = backend_node_id
                    interactable_cleansed_elements.append(cleansed_element)
                    interactable_map[backend_node_id] = cleansed_element
                    if backend_node_id == node_id:
                        current_sublist.append(cleansed_element)
                        long_current_sublist.append(cleansed_element)
                        current_tokens += len(ENCODING.encode(cleansed_element))
                        long_current_tokens += len(ENCODING.encode(cleansed_element))
                if random() < 0.3:
                    random_limit = randrange(500, 3000)
                    long_random_limit = randrange(9000, 12000)
                else:
                    random_limit = 3000
                    long_random_limit = 12000
                for cleansed_element in interactable_cleansed_elements:
                    encoded_element = ENCODING.encode(cleansed_element)
                    cleansed_element = cleansed_element
                    if current_tokens + len(encoded_element) <= random_limit:
                        current_sublist.append(cleansed_element)
                        neg_current_sublist.append(cleansed_element)
                        current_tokens += len(encoded_element)
                    if long_current_tokens + len(encoded_element) <= long_random_limit:
                        long_current_sublist.append(cleansed_element)
                        neg_long_current_sublist.append(cleansed_element)
                        long_current_tokens += len(encoded_element)
                backend_node_id = selected_element['backend_node_id']
                
                del selected_element.attrs['backend_node_id']
                selected_element_str = clean_element(selected_element.prettify())
                selected_element['backend_node_id'] = backend_node_id
                if node_id not in interactable_map:
                    interactable_map[node_id] = selected_element_str
                    interactables.append(interactable_map[node_id])
                indices, current_sublist = shuffle_list_by_indices(current_sublist)
                tag_id = indices.index(0) + 1
                indices, long_current_sublist = shuffle_list_by_indices(long_current_sublist)
                long_tag_id = indices.index(0) + 1
                formatted_items = ['({}) {}'.format(i + 1, element) for i, element in enumerate(current_sublist)]
                elements_str = '\n'.join(formatted_items)
                formatted_items = ['({}) {}'.format(i + 1, element) for i, element in enumerate(long_current_sublist)]
                long_elements_str = '\n'.join(formatted_items)
                formatted_items = ['({}) {}'.format(i + 1, element) for i, element in enumerate(neg_current_sublist)]
                neg_elements_str = '\n'.join(formatted_items)
                formatted_items = ['({}) {}'.format(i + 1, element) for i, element in enumerate(neg_long_current_sublist)]
                neg_long_elements_str = '\n'.join(formatted_items)
                actor.update(interactables=interactables, html=action['raw_html'])
                page_context = actor.get_page_context()
                high_level_element = actor.get_high_level_element(selected_element_str, selected_action.lower())

                if secondary_parameter:
                    candidate = '{action} {text_field} in {element}'.format(action=selected_action.lower(), element=high_level_element, text_field=secondary_parameter)
                    candidate_without_param = '{action} in {element}'.format(action=selected_action.lower(), element=high_level_element)
                else:
                    candidate = '{action} {element}'.format(action=selected_action.lower(), element=high_level_element)
                actor.ACTIONS.append(candidate)
                neg_node = choice(list(interactable_map.keys()))
                while neg_node == node_id:
                    neg_node = choice(list(interactable_map.keys()))
                neg_element = interactable_map[neg_node]
                if neg_element.strip().startswith('<input') or neg_element.strip().startswith('<textarea') or 'role="textbox"' in neg_element:
                    if random() < 0.1:
                        neg_action = 'click'
                    else:
                        neg_action = choice(['type_text', 'type_and_enter', 'press_enter'])
                elif neg_element.strip().startswith('<select'):
                    neg_action = 'select_option'
                else:
                    neg_action = 'click'
                high_level_neg = actor.get_high_level_element(neg_element, neg_action)
                neg_candidate = '{action} {element}'.format(action=neg_action, element=high_level_neg)
                neg_secondary = None
                if neg_action == 'type_text' or neg_action == 'type_and_enter':
                    neg_secondary = actor.text_fields(high_level_neg, neg_action)
                if neg_action == 'select_option':
                    # options = parser.soup.find('select', {'backend_node_id': neg_node}).find_all('option')
                    options = parser.soup.select_one('[backend_node_id="{}"]'.format(node_id)).find_all('option')
                    neg_clean_options = []
                    for option in options:
                        neg_clean_options.append(clean_element(option.prettify()))
                    neg_secondary = actor.select_options(high_level_neg, neg_action, neg_clean_options)
                if neg_secondary:
                    neg_candidate = '{action} {text_field} in {element}'.format(action=neg_action, element=high_level_neg, text_field=neg_secondary)

                makes_sense_pos, _ = OAI.handle_response(prompts.SYS_GENERATE_MAKES_SENSE_YES, prompts.USER_MAKES_SENSE.format(site=website, context=page_context, actions=actor.ACTIONS, candidate=candidate, goal=task))
                if not makes_sense_pos.lower().startswith('yes'):
                    makes_sense_pos, _ = OAI.handle_response(prompts.SYS_GENERATE_MAKES_SENSE_YES, prompts.USER_MAKES_SENSE.format(site=website, context=page_context, actions=actor.ACTIONS, candidate=candidate, goal=task))
                    if not makes_sense_pos.lower().startswith('yes'):
                        makes_sense_pos = 'Yes. {}'.format(makes_sense_pos)

                makes_sense_neg, _ = OAI.handle_response(prompts.SYS_GENERATE_MAKES_SENSE_NO, prompts.USER_MAKES_SENSE.format(site=website, context=page_context, actions=actor.ACTIONS, candidate=neg_candidate, goal=task))
                if not makes_sense_neg.lower().startswith('no'):
                    makes_sense_neg, _ = OAI.handle_response(prompts.SYS_GENERATE_MAKES_SENSE_NO, prompts.USER_MAKES_SENSE.format(site=website, context=page_context, actions=actor.ACTIONS, candidate=neg_candidate, goal=task))
                    if not makes_sense_neg.lower().startswith('no'):
                        makes_sense_neg = 'No. {}'.format(makes_sense_neg)

                if action_id == len(item['actions']) - 1:
                    end_state_label = 'pos'
                    end_state, _ = OAI.handle_response(prompts.SYS_GENERATE_END_STATE_YES, prompts.USER_END_STATE.format(site=website, context=page_context, actions=actor.ACTIONS, goal=task))
                    if not end_state.lower().startswith('yes'):
                        end_state, _ = OAI.handle_response(prompts.SYS_GENERATE_END_STATE_YES, prompts.USER_END_STATE.format(site=website, context=page_context, actions=actor.ACTIONS, goal=task))
                        if not end_state.lower().startswith('yes'):
                            end_state = 'Yes. {}'.format(end_state)
                else:
                    end_state_label = 'neg'
                    end_state, _ = OAI.handle_response(prompts.SYS_GENERATE_END_STATE_NO, prompts.USER_END_STATE.format(site=website, context=page_context, actions=actor.ACTIONS, goal=task))
                    if not end_state.lower().startswith('no'):
                        end_state, _ = OAI.handle_response(prompts.SYS_GENERATE_END_STATE_NO, prompts.USER_END_STATE.format(site=website, context=page_context, actions=actor.ACTIONS, goal=task))
                        if not end_state.lower().startswith('no'):
                            end_state = 'No. {}'.format(end_state)

                if selected_action == 'select_option':
                    # options = parser.soup.find({'backend_node_id': node_id}).find_all('option')
                    options = parser.soup.select_one('[backend_node_id="{}"]'.format(node_id)).find_all('option')
                    clean_options = []
                    matching_option_idx = 1
                    for idx, option in enumerate(options):
                        if ('value' in option.attrs and secondary_parameter.lower() in option.attrs['value'].lower()) or (secondary_parameter.lower() in option.text.lower()):
                            matching_option_idx = idx + 1
                        backend_node_id = option.attrs['backend_node_id']
                        del option.attrs['backend_node_id']
                        clean_options.append(clean_element(option.prettify()))
                        option.attrs['backend_node_id'] = backend_node_id
                    formatted_items = ['({}) {}'.format(i + 1, item) for i, item in enumerate(clean_options)]
                    options_str = '\n'.join(formatted_items)
                
                candidate_with_element = '{candidate}, {element}'.format(candidate=candidate, element=selected_element_str)
                reasoning_pos, _ = OAI.handle_response(prompts.SYS_GENERATE_ACTION_ELEMENT_YES, prompts.USER_ACTION_ELEMENT_YES.format(site=website, context=page_context, actions=actor.ACTIONS[:-1], candidate=candidate_with_element, goal=task))
                reasoning_neg, _ = OAI.handle_response(prompts.SYS_GENERATE_ACTION_ELEMENT_NO, prompts.USER_ACTION_ELEMENT_NO.format(site=website, context=page_context, actions=actor.ACTIONS[:-1], elements=neg_elements_str, goal=task))
                reasoning_rank_pos, _ = OAI.handle_response(prompts.SYS_GENERATE_ACTION_ELEMENT_RANK_YES, prompts.USER_ACTION_ELEMENT_YES.format(site=website, context=page_context, actions=actor.ACTIONS[:-1], candidate=candidate_with_element, goal=task))
                reasoning_rank_neg, _ = OAI.handle_response(prompts.SYS_GENERATE_ACTION_ELEMENT_RANK_NO, prompts.USER_ACTION_ELEMENT_RANK_NO.format(site=website, context=page_context, actions=actor.ACTIONS[:-1], candidates=neg_elements_str, goal=task))
                
                element_list = []
                neg_element_list = []
                temp_elements = long_current_sublist.copy()
                element_list.append(selected_element_str)
                for _ in range(0, 3):
                    formatted_items = ['({}) {}'.format(i + 1, item) for i, item in enumerate(temp_elements)]
                    temp_elements_str = '\n'.join(formatted_items)
                    message, _ = OAI_LONG.handle_response(prompts.SYS_ACTION_ELEMENT, prompts.USER_ACTION_ELEMENT.format(site=website, context=page_context, goal=task, actions=actor.ACTIONS[:-1], elements=temp_elements_str))
                    numbers = re.findall(r'\d+', message)
                    if len(numbers) > 0:
                        idx = int(numbers[0]) - 1
                        if idx < len(temp_elements):
                            if temp_elements[idx] not in element_list:
                                element_list.append(temp_elements[idx])
                                neg_element_list.append(temp_elements[idx])
                                temp_elements.pop(idx)
                indices, element_list = shuffle_list_by_indices(element_list)
                short_tag_id = indices.index(0) + 1
                formatted_items = ['({}) {}'.format(i + 1, element) for i, element in enumerate(element_list)]
                short_elements_str = '\n'.join(formatted_items)
                formatted_items = ['({}) {}'.format(i + 1, element) for i, element in enumerate(neg_element_list)]
                neg_short_elements_str = '\n'.join(formatted_items)
                candidate_with_index = 'ELEMENT ({})'.format(tag_id)
                long_candidate_with_index = 'ELEMENT ({})'.format(long_tag_id)
                short_candidate_with_index = '{} ({})'.format(selected_action, indices.index(0) + 1)
                # print(selected_action, candidate_with_index, current_sublist[tag_id - 1])
                
                candidate_proposals.append({'id': action['action_uid'], 'label': 'pos', 'instruction': choice(prompts.GEN_ACTION_ELEMENT), 'input': prompts.USER_ACTION_ELEMENT.format(site=website, context=page_context, goal=task, actions=actor.ACTIONS[:-1], elements=elements_str), 'output': '{}\n{}'.format(candidate_with_index, reasoning_pos)})
                candidate_proposals.append({'id': action['action_uid'], 'label': 'neg', 'instruction': choice(prompts.GEN_ACTION_ELEMENT), 'input': prompts.USER_ACTION_ELEMENT_NO.format(site=website, context=page_context, goal=task, actions=actor.ACTIONS[:-1], elements=neg_elements_str), 'output': 'None\n{}'.format(reasoning_neg)})
                candidate_proposals_long.append({'id': action['action_uid'], 'label': 'pos', 'instruction': choice(prompts.GEN_ACTION_ELEMENT), 'input': prompts.USER_ACTION_ELEMENT.format(site=website, context=page_context, goal=task, actions=actor.ACTIONS[:-1], elements=long_elements_str), 'output': '{}\n{}'.format(long_candidate_with_index, reasoning_pos)})
                candidate_proposals_long.append({'id': action['action_uid'], 'label': 'neg', 'instruction': choice(prompts.GEN_ACTION_ELEMENT), 'input': prompts.USER_ACTION_ELEMENT_NO.format(site=website, context=page_context, goal=task, actions=actor.ACTIONS[:-1], elements=neg_long_elements_str), 'output': 'None\n{}'.format(reasoning_neg)})
                candidate_action_selections.append({'id': action['action_uid'], 'label': 'pos', 'instruction': choice(prompts.GEN_ACTION_ELEMENT_RANK), 'input': prompts.USER_ACTION_ELEMENT_RANK.format(site=website, context=page_context, goal=task, actions=actor.ACTIONS[:-1], candidates=short_elements_str), 'output': '{}\n{}'.format(short_candidate_with_index, reasoning_rank_pos)})
                candidate_action_selections.append({'id': action['action_uid'], 'label': 'neg', 'instruction': choice(prompts.GEN_ACTION_ELEMENT_RANK), 'input': prompts.USER_ACTION_ELEMENT_RANK_NO.format(site=website, context=page_context, goal=task, actions=actor.ACTIONS[:-1], candidates=neg_short_elements_str), 'output': 'None\n{}'.format(reasoning_rank_neg)})
                end_state_terminations.append({'id': action['action_uid'], 'label': end_state_label, 'instruction': choice(prompts.GEN_END_STATE), 'input': prompts.USER_END_STATE.format(site=website, context=page_context, goal=task, actions=actor.ACTIONS), 'output': end_state})
                makes_senses.append({'id': action['action_uid'], 'label': 'pos', 'instruction': choice(prompts.GEN_MAKES_SENSE), 'input': prompts.USER_MAKES_SENSE.format(site=website, context=page_context, goal=task, actions=actor.ACTIONS[:-1], candidate=candidate), 'output': makes_sense_pos})
                makes_senses.append({'id': action['action_uid'], 'label': 'neg', 'instruction': choice(prompts.GEN_MAKES_SENSE), 'input': prompts.USER_MAKES_SENSE.format(site=website, context=page_context, goal=task, actions=actor.ACTIONS[:-1], candidate=neg_candidate), 'output': makes_sense_neg})
                page_contexts.append({'id': action['action_uid'], 'instruction': choice(prompts.GEN_PAGE_TEXT), 'input': prompts.USER_CONTEXT.format(site=website, page_text=actor.page_text), 'output': page_context})
                if secondary_parameter and selected_action in ['type_text', 'type_and_enter']:
                    text_field, _ = OAI.handle_response(prompts.SYS_TEXT_TYPE, prompts.USER_TEXT_TYPE.format(site=website, context=page_context, candidate=candidate, goal=task))
                    if text_field in TYPEDICT:
                        secondary_parameters.append({'id': action['action_uid'], 'instruction': prompts.SYS_TEXT_TYPE, 'input': prompts.USER_TEXT_TYPE.format(site=website, context=page_context, goal=task, candidate=candidate_without_param), 'output': text_field})
                    else:
                        secondary_parameters.append({'id': action['action_uid'], 'instruction': choice(prompts.GEN_TEXT_FIELD), 'input': prompts.USER_GENERATE_TEXT.format(site=website, context=page_context, goal=task, candidate=candidate_without_param), 'output': secondary_parameter})
                elif secondary_parameter and selected_action == 'select_option':
                    secondary_parameters.append({'id': action['action_uid'], 'instruction': choice(prompts.GEN_SELECT_OPTION), 'input': prompts.USER_GENERATE_SELECT.format(site=website, context=page_context, goal=task, candidate=candidate_without_param, options=options_str, actions=actor.ACTIONS), 'output': matching_option_idx})
                high_level_elements.append({'id': action['action_uid'], 'instruction': choice(prompts.GEN_HIGH_LEVEL), 'input': prompts.USER_HIGH_LEVEL_ACTION.format(site=website, context=page_context, candidate=selected_element), 'output': '{} ({})'.format(high_level_element, selected_element.text)})

                with open('../alpaca_datasets/alpaca2web/{}/candidate_proposal.json'.format(file.replace('.json', '')), 'w') as outfile:
                    json.dump(candidate_proposals, outfile, indent=4)
                with open('../alpaca_datasets/alpaca2web/{}/candidate_proposal_long.json'.format(file.replace('.json', '')), 'w') as outfile:
                    json.dump(candidate_proposals_long, outfile, indent=4)
                with open('../alpaca_datasets/alpaca2web/{}/candidate_action_selection.json'.format(file.replace('.json', '')), 'w') as outfile:
                    json.dump(candidate_action_selections, outfile, indent=4)
                with open('../alpaca_datasets/alpaca2web/{}/end_state_termination.json'.format(file.replace('.json', '')), 'w') as outfile:
                    json.dump(end_state_terminations, outfile, indent=4)
                with open('../alpaca_datasets/alpaca2web/{}/makes_sense.json'.format(file.replace('.json', '')), 'w') as outfile:
                    json.dump(makes_senses, outfile, indent=4)
                with open('../alpaca_datasets/alpaca2web/{}/page_context.json'.format(file.replace('.json', '')), 'w') as outfile:
                    json.dump(page_contexts, outfile, indent=4)
                with open('../alpaca_datasets/alpaca2web/{}/secondary_parameter.json'.format(file.replace('.json', '')), 'w') as outfile:
                    json.dump(secondary_parameters, outfile, indent=4)
                with open('../alpaca_datasets/alpaca2web/{}/high_level_element.json'.format(file.replace('.json', '')), 'w') as outfile:
                    json.dump(high_level_elements, outfile, indent=4)
                done_annotations[item['annotation_id']].append(action['action_uid'])
                with open('../alpaca_datasets/alpaca2web/{}/done_annotations.json'.format(file.replace('.json', '')), 'w') as outfile:
                    json.dump(done_annotations, outfile, indent=4)
        except Exception as e:
            if item['annotation_id'] not in skip_annotations:
                skip_annotations[item['annotation_id']] = []
            skip_annotations[item['annotation_id']].append(action['action_uid'])
            with open('../alpaca_datasets/alpaca2web/{}/skip_annotations.json'.format(file.replace('.json', '')), 'w') as outfile:
                json.dump(skip_annotations, outfile, indent=4)
            print(e)
            continue

# Convert prompts.py to a more readable text file or JSON and then load in, fix small typos and formatting errors
# List of candidates with two different context lengths
# somehow add visiting urls, opening tabs, selecting options, right clicking, uploading, highlighting, copying, pasting, etc.
    # list of select options
    # give explicit file to upload
    # entire page text as input options for highlight, copy, and paste
    # generate url to visit
    # list of tab options
        # open, close, switch to

