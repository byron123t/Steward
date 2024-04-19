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

sample_list = json.load(open('new_annotations.json', 'r'))

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

for file in files:
    if file.startswith('test_domain'):
        newfile = 'test_domain'
    elif file.startswith('test_task'):
        continue
        newfile = 'test_task'
    elif file.startswith('test_website'):
        continue
        newfile = 'test_website'
    else:
        continue
    
    if not os.path.exists('data/results/mind2web/{}_elements.json'.format(newfile.replace('.json', ''))):
        with open('data/results/mind2web/{}_elements.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(elements_dict, outfile, indent=4)
    else:
        with open('data/results/mind2web/{}_elements.json'.format(newfile.replace('.json', '')), 'r') as infile:
            elements_dict = json.load(infile)
    if not os.path.exists('data/results/mind2web/{}_tokens.json'.format(newfile.replace('.json', ''))):
        with open('data/results/mind2web/{}_tokens.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(tokens, outfile, indent=4)
    else:
        with open('data/results/mind2web/{}_tokens.json'.format(newfile.replace('.json', '')), 'r') as infile:
            tokens = json.load(infile)
    if not os.path.exists('data/results/mind2web/{}_scores.json'.format(newfile.replace('.json', ''))):
        with open('data/results/mind2web/{}_scores.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(scores, outfile, indent=4)
    else:
        with open('data/results/mind2web/{}_scores.json'.format(newfile.replace('.json', '')), 'r') as infile:
            scores = json.load(infile)
    if not os.path.exists('data/results/mind2web/{}_proportions.json'.format(newfile.replace('.json', ''))):
        with open('data/results/mind2web/{}_proportions.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(proportions, outfile, indent=4)
    else:
        with open('data/results/mind2web/{}_proportions.json'.format(newfile.replace('.json', '')), 'r') as infile:
            proportions = json.load(infile)
    if not os.path.exists('data/results/mind2web/{}_task_proportions.json'.format(newfile.replace('.json', ''))):
        with open('data/results/mind2web/{}_task_proportions.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(task_proportions, outfile, indent=4)
    else:
        with open('data/results/mind2web/{}_task_proportions.json'.format(newfile.replace('.json', '')), 'r') as infile:
            task_proportions = json.load(infile)
    
    if not os.path.exists('../alpaca_datasets/alpaca2web_test/{}/'.format(newfile.replace('.json', ''))):
        os.mkdir('../alpaca_datasets/alpaca2web_test/{}/'.format(newfile.replace('.json', '')))
        with open('../alpaca_datasets/alpaca2web_test/{}/process_screenshots.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(next_actions, outfile, indent=4)
        with open('../alpaca_datasets/alpaca2web_test/{}/high_level_element.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(high_level_elements, outfile, indent=4)
        with open('../alpaca_datasets/alpaca2web_test/{}/limiter_strings.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(limiter_strings, outfile, indent=4)
        with open('../alpaca_datasets/alpaca2web_test/{}/candidate_proposal.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(candidate_proposals, outfile, indent=4)
        with open('../alpaca_datasets/alpaca2web_test/{}/candidate_action_selection.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(candidate_action_selections, outfile, indent=4)
        with open('../alpaca_datasets/alpaca2web_test/{}/end_state_termination.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(end_state_terminations, outfile, indent=4)
        with open('../alpaca_datasets/alpaca2web_test/{}/page_context.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(page_contexts, outfile, indent=4)
        with open('../alpaca_datasets/alpaca2web_test/{}/secondary_parameter.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(secondary_parameters, outfile, indent=4)
        with open('../alpaca_datasets/alpaca2web_test/{}/done_annotations.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(done_annotations, outfile, indent=4)
        with open('../alpaca_datasets/alpaca2web_test/{}/skip_annotations.json'.format(newfile.replace('.json', '')), 'w') as outfile:
            json.dump(skip_annotations, outfile, indent=4)
    else:
        with open('../alpaca_datasets/alpaca2web_test/{}/process_screenshots.json'.format(newfile.replace('.json', '')), 'r') as infile:
            next_actions = json.load(infile)
        with open('../alpaca_datasets/alpaca2web_test/{}/high_level_element.json'.format(newfile.replace('.json', '')), 'r') as infile:
            high_level_elements = json.load(infile)
        with open('../alpaca_datasets/alpaca2web_test/{}/limiter_strings.json'.format(newfile.replace('.json', '')), 'r') as infile:
            limiter_strings = json.load(infile)
        with open('../alpaca_datasets/alpaca2web_test/{}/candidate_proposal.json'.format(newfile.replace('.json', '')), 'r') as outfile:
            candidate_proposals = json.load(outfile)
        with open('../alpaca_datasets/alpaca2web_test/{}/candidate_action_selection.json'.format(newfile.replace('.json', '')), 'r') as outfile:
            candidate_action_selections = json.load(outfile)
        with open('../alpaca_datasets/alpaca2web_test/{}/end_state_termination.json'.format(newfile.replace('.json', '')), 'r') as outfile:
            end_state_terminations = json.load(outfile)
        with open('../alpaca_datasets/alpaca2web_test/{}/page_context.json'.format(newfile.replace('.json', '')), 'r') as outfile:
            page_contexts = json.load(outfile)
        with open('../alpaca_datasets/alpaca2web_test/{}/secondary_parameter.json'.format(newfile.replace('.json', '')), 'r') as outfile:
            secondary_parameters = json.load(outfile)
        with open('../alpaca_datasets/alpaca2web_test/{}/done_annotations.json'.format(newfile.replace('.json', '')), 'r') as infile:
            done_annotations = json.load(infile)
        with open('../alpaca_datasets/alpaca2web_test/{}/skip_annotations.json'.format(newfile.replace('.json', '')), 'r') as infile:
            skip_annotations = json.load(infile)

    with open('../alpaca_datasets/Mind2Web/data/test/{}'.format(file), 'r') as infile:
        data = json.load(infile)
        for item in tqdm(data):
            if EXIT_BOOL:
                exit(0)
            if item['annotation_id'] not in sample_list[newfile][:30]:
                continue
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
            # screenshot = json.load(open('../alpaca_datasets/mind2web_screenshots/{}.json'.format(item['annotation_id']), 'r'))
            actor = PageStateActor(website, website, task, interactables, item['actions'][0]['raw_html'], None, None, 12000, None, None, False)
            scores['total_tasks'] += 1
            scores['cur_correct_actions'] = 0
            scores['cur_total_actions'] = len(item['actions'])
            
            progress_bar = tqdm(enumerate(item['actions']), total=len(item['actions']))
            try:
                for action_id, action in progress_bar:
                    if item['annotation_id'] in done_annotations:
                        if action['action_uid'] in done_annotations[item['annotation_id']]:
                            continue
                    selected_action = ACTION_VERB_MAP[action['operation']['op'].lower()]
                    if selected_action not in ['click', 'type_text', 'select_option']:
                        scores['cur_total_actions'] -= 1
                        continue
                    
                    scores['total_actions'] += 1
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
                    scores['task_total_actions'] += 1

                    actor.update(interactables=interactables, html=action['raw_html'], image_bytes=None, image_path='data/screenshots/mind2web/{}/{}.jpeg'.format(item['annotation_id'], action['action_uid']))
                    next_action, input_tokens, output_tokens = actor.process_screenshot(encoded=False)
                    actor.next_action = next_action
                    tokens['process_screenshot']['input'].append(input_tokens)
                    tokens['process_screenshot']['output'].append(output_tokens)
                    print(website, task, item['annotation_id'])
                    print(next_action)
                    print(action_representations)
                    if not next_action.lower().startswith('type') and not next_action.lower().startswith('click') and not next_action.lower().startswith('select') and not next_action.lower().startswith('press') and not next_action.lower().startswith('upload'):
                        continue

                    selected_element = parser.soup.select_one('[backend_node_id="{}"]'.format(node_id))

                    if selected_action.startswith('select'):
                        try:
                            options = parser.soup.select_one('[backend_node_id="{}"]'.format(node_id)).find_all('option')
                        except Exception as e:
                            options = None

                    interactables = parser.get_interactables(clean=False)
                    is_in_interactables = False
                    for i, element in enumerate(interactables):
                        if 'backend_node_id' in element.attrs:
                            if element.attrs['backend_node_id'] == node_id:
                                is_in_interactables = True
                                true_element_tag_id = i
                            backend_node_id = element.attrs['backend_node_id']
                    if is_in_interactables:
                        elements_dict['is_in_interactables'].append(True)
                    else:
                        interactables.append(selected_element)
                        true_element_tag_id = len(interactables) - 1
                        elements_dict['is_in_interactables'].append(False)

                    if 'backend_node_id' in selected_element.attrs:
                        del selected_element.attrs['backend_node_id']
                    if 'data_pw_testid_buckeye' in selected_element.attrs:
                        del selected_element.attrs['data_pw_testid_buckeye']
                    selected_element_str = clean_element(selected_element.prettify()).lower().strip()
                    print(selected_element_str)

                    # high_level_element = ' '.join(next_action.split()[1:])
                    actor.context, input_tokens, output_tokens = actor.get_page_context()
                    tokens['page_context']['input'].append(input_tokens)
                    tokens['page_context']['output'].append(output_tokens)
                    high_level_element = actor.get_high_level_element(selected_element_str)
                    
                    interactables = parser.get_interactables(clean=True)
                    
                    sublists, filtered_html, cleaned_html, limited_html, input_tokens, output_tokens, has_true_element, true_tag_id, limit_strings = filter_elements(actor, interactables, selected_element.prettify().lower().strip())
                    
                    total_elements = 0
                    for sublist in sublists:
                        total_elements += len(sublist)
                    if len(sublists) == 0 or total_elements == 0:
                        continue
                    
                    elements_dict['is_in_limited_elements'].append(has_true_element)
                    
                    tokens['raw_html'].append(len(ENCODING.encode(action['raw_html'])))
                    tokens['mind2web_html'].append(len(ENCODING.encode(action['cleaned_html'])))
                    tokens['filtered_html'].append(len(ENCODING.encode(filtered_html)))
                    tokens['clean_html'].append(len(ENCODING.encode(cleaned_html)))
                    tokens['limited_html'].append(len(ENCODING.encode(limited_html)))
                    tokens['limit_elements']['input'].append(input_tokens)
                    tokens['limit_elements']['output'].append(output_tokens)
                    
                    newsublists = []
                    current_sublist = []
                        
                    correct_dict = {'element_proposal': False, 'action_element_selection': False, 'makes_sense': False, 'end_state': False, 'secondary_parameter': False}
                    
                    tag_id, tag_id_list, newelement, cur_action, verb, message, tokens_temp = actor.next_element(sublists)
                    
                    element_candidates = []
                    for sublist in sublists:
                        for tag in tag_id_list:
                            element_candidates.append(sublist[tag])
                    
                    print(selected_element_str)
                    
                    found_element_top_15 = False
                    matches = get_close_matches(selected_element_str, element_candidates, n=1, cutoff=0.95)
                    if len(matches) > 0:
                        found_element_top_15 = True
                    
                    # for element in element_candidates:
                    #     print(element)
                    
                    if (true_tag_id and int(true_tag_id) in tag_id_list) or found_element_top_15:
                        scores['element_proposal_match'] += 1
                        correct_dict['element_proposal'] = True
                    else:
                        print(true_tag_id, tag_id_list)
                    print(true_tag_id, tag_id_list, newelement, selected_element_str)
                    
                    found_element = False
                    matches = get_close_matches(selected_element_str, [newelement], n=1, cutoff=0.95)
                    if len(matches) > 0:
                        found_element = True
                        
                    print(found_element_top_15, found_element, selected_element_str, newelement)
                    

                    action_correct = False
                    close_correct = False
                    if verb == selected_action and (tag_id == true_tag_id or found_element):
                        correct_dict['action_element_selection'] = True
                        scores['action_element_selection_match'] += 1
                        scores['action_element_selection_match_close'] += 1
                        print(selected_action, verb, selected_element_str, newelement)
                    if verb == selected_action:
                        scores['action_selection_match'] += 1
                        action_correct = True
                    if tag_id == true_tag_id or found_element:
                        scores['element_selection_match'] += 1
                        scores['element_selection_match_close'] += 1
                    else:
                        if gpt_score_element(selected_element_str, newelement):
                            if action_correct:
                                scores['action_element_selection_match_close'] += 1
                                close_correct = True
                            scores['element_selection_match_close'] += 1
                    scores['action_element_selection_total'] += 1
                    
                    for key, value in tokens_temp.items():
                        if key not in tokens:
                            tokens[key] = {'input': [], 'output': []}
                        tokens[key]['input'].append(tokens_temp[key]['input'])
                        tokens[key]['output'].append(tokens_temp[key]['output'])


                    text_field_censored = None
                    text_field = None
                    select_option = None
                    high_level_action = '{} {}'.format(verb, high_level_element)
                    if verb.startswith('type'):
                        text_field_message, text_field, text_field_censored, input_tokens, output_tokens = actor.text_fields(high_level_action, verb)
                        tokens['text_field']['input'].append(input_tokens)
                        tokens['text_field']['output'].append(output_tokens)
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
                                        scores['text_field_match_close'] += 1
                                    else:
                                        print(secondary_parameter, text_field_censored)
                            scores['text_field_total'] += 1
                    elif verb.startswith('select'):
                        if selected_action.startswith('select_option'):
                            if options:
                                select_option_message, select_option, input_tokens, output_tokens = actor.select_options(high_level_action, verb, options)
                                tokens['select_option']['input'].append(input_tokens)
                                tokens['select_option']['output'].append(output_tokens)
                                if select_option:
                                    high_level_action = '{} {} in {}'.format(verb, select_option, high_level_element)
                                if select_option == secondary_parameter:
                                    correct_dict['secondary_parameter'] = True
                                    scores['select_option_match'] += 1
                                else:
                                    print(secondary_parameter, select_option)
                                scores['select_option_total'] += 1
                    else:
                        correct_dict['secondary_parameter'] = True

                    if correct_dict['action_element_selection'] and correct_dict['secondary_parameter']:
                        scores['correct_actions'] += 1
                        scores['cur_correct_actions'] += 1
                        # scores['correct_proportion'] = scores['correct_actions'] / scores['total_actions']
                        scores['task_correct_actions'] += 1
                        scores['correct_proportion'] = scores['task_correct_actions'] / scores['task_total_actions']

                    if secondary_parameter:
                        candidate = '{action} {text_field} in {element}'.format(action=selected_action.lower(), element=high_level_element, text_field=secondary_parameter)
                        candidate_without_param = '{action} in {element}'.format(action=selected_action.lower(), element=high_level_element)
                        if selected_action.lower().startswith('type'):
                            selected_candidate = '{action} {text_field} in {element}'.format(action=verb.lower(), element=newelement, text_field=text_field_censored)
                        else:
                            selected_candidate = '{action} {option} in {element}'.format(action=verb.lower(), element=newelement, option=select_option)
                    else:
                        candidate = '{action} {element}'.format(action=selected_action.lower(), element=high_level_element)
                        selected_candidate = '{action} {element}'.format(action=verb.lower(), element=newelement)
                    print(selected_candidate)
                    actor.ACTIONS.append(candidate)

                    end_state, end_state_message, input_tokens, output_tokens = actor.end_state()
                    tokens['end_state']['input'].append(input_tokens)
                    tokens['end_state']['output'].append(output_tokens)
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
                    
                    proportions.append(scores['correct_proportion'])
                    if scores['correct_proportion'] == 1.0:
                        scores['correct_tasks'] += 1
                    task_proportions.append(scores['correct_tasks'] / scores['total_tasks'])
                    # print(selected_action, candidate_with_index, current_sublist[tag_id - 1])
                    next_actions.append({'annotation_id': item['annotation_id'], 'action_uid': action['action_uid'], 'next_action': next_action})
                    high_level_elements.append({'annotation_id': item['annotation_id'], 'action_uid': action['action_uid'], 'high_level_element': high_level_element})
                    limiter_strings.append({'annotation_id': item['annotation_id'], 'action_uid': action['action_uid'], 'limiter_string': limit_strings})
                    candidate_proposals.append({'annotation_id': item['annotation_id'], 'action_uid': action['action_uid'], 'candidate_proposal': element_candidates, 'gt_action': selected_action, 'gt_element': selected_element_str})
                    candidate_action_selections.append({'annotation_id': item['annotation_id'], 'action_uid': action['action_uid'], 'action': verb, 'element': newelement, 'gt_action': selected_action, 'gt_element': selected_element_str})
                    end_state_terminations.append({'annotation_id': item['annotation_id'], 'action_uid': action['action_uid'], 'end_state': end_state})
                    page_contexts.append({'annotation_id': item['annotation_id'], 'action_uid': action['action_uid'], 'page_context': actor.context})
                    secondary_parameters.append({'annotation_id': item['annotation_id'], 'action_uid': action['action_uid'], 'text_field': text_field_censored, 'select_option': select_option})
                    
                    with open('../alpaca_datasets/alpaca2web_test/{}/process_screenshots.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(next_actions, outfile, indent=4)
                    with open('../alpaca_datasets/alpaca2web_test/{}/high_level_element.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(high_level_elements, outfile, indent=4)
                    with open('../alpaca_datasets/alpaca2web_test/{}/limiter_strings.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(limiter_strings, outfile, indent=4)
                    with open('../alpaca_datasets/alpaca2web_test/{}/candidate_proposal.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(candidate_proposals, outfile, indent=4)
                    with open('../alpaca_datasets/alpaca2web_test/{}/candidate_action_selection.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(candidate_action_selections, outfile, indent=4)
                    with open('../alpaca_datasets/alpaca2web_test/{}/end_state_termination.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(end_state_terminations, outfile, indent=4)
                    with open('../alpaca_datasets/alpaca2web_test/{}/page_context.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(page_contexts, outfile, indent=4)
                    with open('../alpaca_datasets/alpaca2web_test/{}/secondary_parameter.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(secondary_parameters, outfile, indent=4)
            
                    done_annotations[item['annotation_id']].append(action['action_uid'])
                    with open('../alpaca_datasets/alpaca2web_test/{}/done_annotations.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(done_annotations, outfile, indent=4)
                    with open('data/results/mind2web/{}_tokens.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(tokens, outfile, indent=4)
                    with open('data/results/mind2web/{}_elements.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(elements_dict, outfile, indent=4)
                    with open('data/results/mind2web/{}_scores.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(scores, outfile, indent=4)
                    with open('data/results/mind2web/{}_proportions.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(proportions, outfile, indent=4)
                    with open('data/results/mind2web/{}_task_proportions.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(task_proportions, outfile, indent=4)
                    with open('data/results/mind2web/{}_scores.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                        json.dump(scores, outfile, indent=4)
                    BM.write_to_file()
            except Exception as e:
                if item['annotation_id'] not in skip_annotations:
                    skip_annotations[item['annotation_id']] = []
                skip_annotations[item['annotation_id']].append(action['action_uid'])
                with open('../alpaca_datasets/alpaca2web_test/{}/skip_annotations.json'.format(newfile.replace('.json', '')), 'w') as outfile:
                    json.dump(skip_annotations, outfile, indent=4)
                print()
                print('============================')
                print('============================')
                print(e)
                print('============================')
                print('============================')
                print()
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

