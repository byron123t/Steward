import re
import os
import json
import random
import argparse
import tiktoken
import playwright
import pandas as pd
from playwright.sync_api import Page, expect
from difflib import get_close_matches
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, datetime, MAXYEAR, MINYEAR
from src.action import *
from src.parse_html import ParseHtml
from src.API import OpenAIAPI, handle_image
import data.runtime.prompts as prompts
from sensitive import *
from src.html_processing import clean_element
from src.benchmark import BM


ACTION_VERB_MAP = {'click': click, 'type_text': type_text, 'drag': drag, 'check_item': check, 'press_enter': enter, 'right_click': right_click, 'select_option': select, 'visit_url': exploration, 'upload_file': upload_file, 'type_and_enter': type_submit}
#  'copy': copy, 'paste': paste

global OAI
# global OAI_LONG
global OAI_GOOD
global OAI_MAKES_SENSE
global OAI_END_STATE
global OAI_IMAGE
global ENCODING
OAI = OpenAIAPI(model='gpt-3.5-turbo-1106')
# OAI_LONG = OpenAIAPI(model='gpt-3.5-turbo-16k')
OAI_GOOD = OpenAIAPI(model='gpt-4-1106-preview')
OAI_IMAGE = OpenAIAPI(model='gpt-4-vision-preview')
OAI_MAKES_SENSE = OpenAIAPI(model='ft:gpt-3.5-turbo-0613:MODEL_NAME_HERE')
OAI_END_STATE = OpenAIAPI(model='ft:gpt-3.5-turbo-0613:MODEL_NAME_HERE')
ENCODING = tiktoken.encoding_for_model('gpt-3.5-turbo-16k')


def cache_timestamp_expiration(items, max_frequency=1000):
    timestamp_key = None
    total_freq = 0
    earliest = datetime(MAXYEAR, 1, 1, 0, 0, 0, 0)
    for key, data in items.items():
        total_freq += data['frequency']
        timestamp = datetime.fromisoformat(data['timestamps'][0])
        if timestamp < earliest:
            earliest = timestamp
            timestamp_key = key
    if total_freq > max_frequency:
        items[timestamp_key].pop(0)
    return items

def cache_replacement_policy(sequences, base_url_only, cache_key, verb, element, mode='actions', replacement='LRU', cache_size=100, max_frequency=1000):
    if len(list(sequences[base_url_only][mode].keys())) < cache_size:
        if mode == 'actions':
            sequences[base_url_only][mode][cache_key] = {'verb': verb, 'element': element, 'timestamps': [datetime.today()], 'frequency': 1}
            # sequences[base_url_only][mode][cache_key] = {'verb': verb, 'element': element, 'prev_state': , 'next_state': , 'timestamps': [datetime.today()]}
        elif mode == 'goals':
            sequences[base_url_only][mode][cache_key] = [{'verb': verb, 'element': element, 'timestamps': [datetime.today()], 'frequency': 1}]
    else:
        delete_key = None
        if replacement == 'LRU':
            smallest = datetime(MAXYEAR)
        elif replacement == 'LFU':
            smallest = 1e10
        for key, data in sequences[base_url_only][mode].items():
            total_freq += data['frequency']
            timestamp = datetime.fromisoformat(data['timestamps'][-1])
            if replacement == 'LRU' and timestamp < smallest:
                smallest = timestamp
                delete_key = key
            elif replacement == 'LFU' and len(data['timestamps'] < smallest):
                smallest = len(data['timestamps'])
                delete_key = key
        del sequences[base_url_only][mode][delete_key]
        sequences[base_url_only][mode] = cache_timestamp_expiration(sequences[base_url_only][mode], max_frequency=max_frequency)
        if mode == 'actions':
            sequences[base_url_only][mode][cache_key] = {'verb': verb, 'element': element , 'timestamps': [datetime.today()], 'frequency': 1}
            # sequences[base_url_only][mode][cache_key] = {'verb': verb, 'element': element, 'prev_state': , 'next_state': , 'timestamps': [datetime.today()]}
        elif mode == 'goals':
            sequences[base_url_only][mode][cache_key].append({'verb': verb, 'element': element, 'timestamps': [datetime.today()], 'frequency': 1})
    return sequences
    

class PageStateActor:
    def __init__(self, base_url:str, base_url_only:str, goal:str, interactables:list, html:str, tab:Page, tabs:list, max_tokens:int, image_path:str, image_bytes: str, verbose:bool=False):
        self.MAX_TOKENS = max_tokens
        self.ACTIONS = []
        self.STATES = []
        self.tab = tab
        self.tabs = tabs
        self.verbose = verbose
        self.base_url = base_url
        self.base_url_only = base_url_only
        self.goal = goal
        self.next_action = None
        self.interactables = interactables
        self.html = html
        # self.context, _ = self.get_page_context()
        self.goal = goal
        self.image_path = image_path
        self.image_bytes = image_bytes
        self.cache_replacement = 'LRU'
        self.cache_size = 100
        self.cache_max_freq = 1000

    def update(self, base_url:str=None, goal:str=None, interactables:list=None, html:str=None, tab:Page=None, tabs:list=None, image_path:str=None, image_bytes:str=None):
        if base_url:
            self.base_url = base_url
        if goal:
            self.goal = goal
        if interactables:
            self.interactables = interactables
        if html:
            self.html = html
            # self.context, _ = self.get_page_context()
        if tab:
            self.tab = tab
        if tabs:
            self.tabs = tabs
        if image_path:
            self.image_path = image_path
        if image_bytes:
            self.image_bytes = image_bytes
    
    def element_proposal(self, sublists: list, n_tries: int=3):
        element_list = []
        tag_id_list = []
        input_token_count = 0
        output_token_count = 0
        current_chunk_len = 0
        current_candidate = 1
        if self.verbose: print('NUMBER OF ELEMENT CHUNKS\n{}\n\n'.format(len(sublists)))
        for chunk, elements in enumerate(sublists):
            if self.verbose: print('ELEMENT CHUNK {}, {} elements\n\n'.format(chunk, len(elements)))
            temp_elements = elements.copy()
            for _ in range(0, n_tries):
                formatted_items = ['({}) {}'.format(i + 1, item) for i, item in enumerate(temp_elements)]
                elements_str = '\n\n'.join(formatted_items)
                input_token_count += len(ENCODING.encode(prompts.USER_ACTION_ELEMENT.format(site=self.base_url, context=self.context, goal=self.goal, next_action=self.next_action, actions=self.ACTIONS, elements=elements_str))) + len(ENCODING.encode(prompts.SYS_ACTION_ELEMENT))
                message, _ = OAI.handle_response(prompts.SYS_ACTION_ELEMENT, prompts.USER_ACTION_ELEMENT.format(site=self.base_url, context=self.context, goal=self.goal, next_action=self.next_action, actions=self.ACTIONS, elements=elements_str))
                output_token_count += len(ENCODING.encode(message))
                numbers = re.findall(r'\d+', message)
                if len(numbers) > 0:
                    idx = int(numbers[0]) - 1
                    if idx < len(temp_elements):
                        tag_id_list.append(elements.index(temp_elements[idx]) + current_chunk_len)
                        element_list.append(temp_elements[idx])
                        if self.verbose: print('CANDIDATE ACTION ELEMENT ({})\n{}\n{}\n\n'.format(current_candidate, message, temp_elements[idx]))
                        temp_elements.pop(idx)
                        sublists[chunk] = temp_elements
                        current_candidate += 1
            current_chunk_len += len(elements)
        return tag_id_list, element_list, input_token_count, output_token_count

    def element_filtering(self, sublists: list):
        element_list = []
        tag_id_list = []
        input_token_count = 0
        output_token_count = 0
        current_chunk_len = 0
        current_candidate = 1
        if self.verbose: print('NUMBER OF ELEMENT CHUNKS\n{}\n\n'.format(len(sublists)))
        for chunk, elements in enumerate(sublists):
            if self.verbose: print('ELEMENT CHUNK {}, {} elements\n\n'.format(chunk, len(elements)))
            temp_elements = elements.copy()
            formatted_items = ['({}) {}'.format(i + 1, item) for i, item in enumerate(temp_elements)]
            elements_str = '\n\n'.join(formatted_items)
            input_token_count += len(ENCODING.encode(prompts.USER_ACTION_ELEMENT_FILTER.format(site=self.base_url, context=self.context, goal=self.goal, next_action=self.next_action, actions=self.ACTIONS, elements=elements_str))) + len(ENCODING.encode(prompts.SYS_ACTION_ELEMENT_FILTER))
            message, _ = OAI.handle_response(prompts.SYS_ACTION_ELEMENT_FILTER, prompts.USER_ACTION_ELEMENT_FILTER.format(site=self.base_url, context=self.context, goal=self.goal, next_action=self.next_action, actions=self.ACTIONS, elements=elements_str))
            output_token_count += len(ENCODING.encode(message))
            numbers = re.findall(r'\d+', message)
            if len(numbers) > 0:
                if self.verbose: print('{}\n{}'.format(numbers, message))
                for number in numbers[:20]:
                    idx = int(number) - 1
                    if idx < len(temp_elements):
                        tag_id_list.append(elements.index(temp_elements[idx]) + current_chunk_len)
                        element_list.append(temp_elements[idx])
                        if self.verbose: print('CANDIDATE ACTION ELEMENT ({})\n{}\n\n'.format(current_candidate, temp_elements[idx]))
                        current_candidate += 1
            current_chunk_len += len(elements)
        return tag_id_list, element_list, input_token_count, output_token_count

    def action_element_selection(self, tag_id_list, element_list):
        print(tag_id_list)
        input_token_count = 0
        output_token_count = 0
        if len(tag_id_list) > 0:
            formatted_items = ['({}) {}'.format(i + 1, item) for i, item in enumerate(element_list)]
            # formatted_items = ['({}) {} {}'.format(i + 1, verb_list[i], item) for i, item in enumerate(element_list)]
            candidates = '\n\n'.join(formatted_items)
            verb = None
            input_token_count = len(ENCODING.encode(prompts.USER_ACTION_ELEMENT_RANK.format(site=self.base_url, context=self.context, goal=self.goal, next_action=self.next_action, actions=self.ACTIONS, candidates=candidates))) + len(ENCODING.encode(prompts.SYS_ACTION_ELEMENT_RANK))
            message, _ = OAI_GOOD.handle_response(prompts.SYS_ACTION_ELEMENT_RANK, prompts.USER_ACTION_ELEMENT_RANK.format(site=self.base_url, context=self.context, goal=self.goal, next_action=self.next_action, actions=self.ACTIONS, candidates=candidates))
            output_token_count += len(ENCODING.encode(message))
            numbers = re.findall(r'\d+', message)
            if len(numbers) > 0:
                index = int(numbers[0]) - 1
                if index < len(tag_id_list):
                    split = message.split(' ')
                    possible_first_verb = split[0].lower()
                    for key in ACTION_VERB_MAP.keys():
                        if key in possible_first_verb:
                            verb = key
                    if not verb:
                        for key in ACTION_VERB_MAP.keys():
                            if key in message.lower():
                                verb = key
                    if not verb:
                        return  None, None, None, None, None, None, None
                    tag_id = tag_id_list[index]
                    element = element_list[index]
                    action = ACTION_VERB_MAP[verb]
                    newelement = element
                    if self.verbose: print('SELECTED ACTION ELEMENT\n{}\n{}\n{}\n\n'.format(verb, index + 1, element))
                else:
                    return None, None, None, None, None, None, None
            else:
                return  None, None, None, None, None, None, None
        else:
            return  None, None, None, None, None, None, None
        return tag_id, newelement, action, verb, message, input_token_count, output_token_count

    def next_element(self, sublists: list, n_tries: int=3):
        tag_id = None
        newelement = None
        element_list = []
        tag_id_list = []
        tokens = {'element_filtering': {'input': 0, 'output': 0}, 'makes_sense': {'input': 0, 'output': 0}, 'action_element_selection': {'input': 0, 'output': 0}}
        current_chunk_len = 0
        temp_sublists = sublists.copy()
        outer_counter = 0
        while outer_counter < n_tries:
            outer_counter += 1
            BM.mark('element_proposal')
            tag_id_list, element_list, tokens['element_filtering']['input'], tokens['element_filtering']['output'] = self.element_filtering(temp_sublists)
            BM.mark('element_proposal')
            BM.mark('makes_sense_checking')
            counter = 0
            makes_sense, _, tokens['makes_sense']['input'], tokens['makes_sense']['output'] = self.makes_sense_elements(element_list)
            while not makes_sense:
                if counter > n_tries:
                    break
                tag_id_list, element_list, tokens['element_filtering']['input'], tokens['element_filtering']['output'] = self.element_filtering(temp_sublists)
                makes_sense, _, tokens['makes_sense']['input'], tokens['makes_sense']['output'] = self.makes_sense_elements(element_list)
                counter += 1
            BM.mark('makes_sense_checking')
            BM.mark('element_action_selection')
            tag_id, newelement, action, verb, message, tokens['action_element_selection']['input'], tokens['action_element_selection']['output'] = self.action_element_selection(tag_id_list, element_list)
            BM.mark('element_action_selection')
            print(tag_id, newelement, action, verb, message)
            if tag_id is not None:
                if self.verbose: print(tag_id, current_chunk_len)
                return tag_id, tag_id_list, newelement, action, verb, message, tokens
            elif verb == 'press_enter':
                return self.tag_id, self.tag_id_list, self.newelement, action, verb, message, tokens
        return None, None, None, None, None, None, tokens

    def text_fields(self, high_level_action: str, verb: str):
        input_token_count = 0
        output_token_count = 0
        if 'type' in verb:
            BM.mark('text_field')
            # text_field_message, _ = OAI.handle_response(prompts.SYS_TEXT_TYPE, prompts.USER_TEXT_TYPE.format(site=self.base_url, context=self.context, goal=self.goal, candidate=high_level_action))
            # text_field = None
            # for key in TYPEDICT.keys():
            #     if key in text_field_message:
            #         text_field = TYPEDICT[key]
            #         text_field_censored = key.upper()
            # if not text_field:
            input_token_count = len(ENCODING.encode(prompts.USER_GENERATE_TEXT.format(site=self.base_url, context=self.context, goal=self.goal, candidate=high_level_action))) + len(ENCODING.encode(prompts.SYS_GENERATE_TEXT))
            text_field_message, _ = OAI_GOOD.handle_response(prompts.SYS_GENERATE_TEXT, prompts.USER_GENERATE_TEXT.format(site=self.base_url, context=self.context, goal=self.goal, candidate=high_level_action))
            output_token_count = len(ENCODING.encode(text_field_message))
            if text_field_message.lower().startswith('type "') or text_field_message.lower().startswith('typed "'):
                matches = re.findall(r'"([^"]*)"', text_field_message)
                if len(matches) > 0:
                    text_field = matches[0].replace('"', '').strip()
                    text_field_censored = text_field
                else:
                    text_field = text_field_message.replace('"', '').strip()
                    text_field_censored = text_field
            else:
                text_field = text_field_message.replace('"', '').strip()
                text_field_censored = text_field
            BM.mark('text_field')
        else:
            text_field_message = None
            text_field = None
            text_field_censored = None
        return text_field_message, text_field, text_field_censored, input_token_count, output_token_count
    
    def select_options(self, high_level_action: str, verb: str, options: list):
        input_token_count = 0
        output_token_count = 0
        BM.mark('select_option')
        if 'select' in verb:
            formatted_items = ['({}) {}'.format(i + 1, element) for i, element in enumerate(options)]
            options_str = '\n\n'.join(formatted_items)
            input_token_count = len(ENCODING.encode(prompts.USER_GENERATE_SELECT.format(site=self.base_url, context=self.context, goal=self.goal, candidate=high_level_action, options=options_str))) + len(ENCODING.encode(prompts.SYS_GENERATE_SELECT))
            select_option_message, _ = OAI.handle_response(prompts.SYS_GENERATE_SELECT, prompts.USER_GENERATE_SELECT.format(site=self.base_url, context=self.context, goal=self.goal, candidate=high_level_action, options=options_str))
            output_token_count = len(ENCODING.encode(select_option_message))
            numbers = re.findall(r'\d+', select_option_message)
            if len(numbers) > 0:
                select_option = int(numbers[0]) - 1
            else:
                select_option = 0
        else:
            select_option_message = None
            select_option = None
            token_count = None
        BM.mark('select_option')
        return select_option_message, select_option, input_token_count, output_token_count

    def get_page_context(self):
        """
        This function gets the page context by parsing HTML and prompting the user for additional
        context.
        :return: the context obtained from the user through the OAI.handle_response() method.
        """
        input_token_count = 0
        output_token_count = 0
        BM.mark('get_page_context')
        input_token_count = len(ENCODING.encode(prompts.USER_CONTEXT.format(site=self.base_url, page_text=self.page_text))) + len(ENCODING.encode(prompts.SYS_CONTEXT))

        context, _ = OAI.handle_response(prompts.SYS_CONTEXT, prompts.USER_CONTEXT.format(site=self.base_url, page_text=self.page_text))
        output_token_count = len(ENCODING.encode(context))
        # if len(encoded_element) < 3500:
        # else:
        #     context, _ = OAI_LONG.handle_response(prompts.SYS_CONTEXT, prompts.USER_CONTEXT.format(site=self.base_url, page_text=self.page_text))
        if self.verbose: print('CONTEXT\n{}\n\n'.format(context))
        BM.mark('get_page_context')
        return context, input_token_count, output_token_count

    def makes_sense_elements(self, elements: list):
        input_token_count = 0
        output_token_count = 0
        input_token_count = len(ENCODING.encode(prompts.USER_MAKES_SENSE_ELEMENTS.format(site=self.base_url, context=self.context, goal=self.goal, actions=self.ACTIONS, candidates=elements, next_action=self.next_action))) + len(ENCODING.encode(prompts.SYS_MAKES_SENSE_ELEMENTS))
        state, _ = OAI_GOOD.handle_response(prompts.SYS_MAKES_SENSE_ELEMENTS, prompts.USER_MAKES_SENSE_ELEMENTS.format(site=self.base_url, context=self.context, goal=self.goal, actions=self.ACTIONS, candidates=elements, next_action=self.next_action))
        output_token_count = len(ENCODING.encode(state))
        stripped_state = state.replace('"', '').replace('.', '').lower().strip()
        if self.verbose: print('CHECKING COMMON SENSE\n{}\n\n'.format(stripped_state))
        if stripped_state.startswith('yes') or stripped_state.endswith('yes'):
            return True, state, input_token_count, output_token_count
        return False, state, input_token_count, output_token_count

    def makes_sense(self, action: str, element: str, text_field: str):
        if text_field:
            candidate = '{action} {text_field} in {element}'.format(action=action, element=element, text_field=text_field)
        else:
            candidate = '{action} {element}'.format(action=action, element=element)
        token_count = len(ENCODING.encode(prompts.USER_MAKES_SENSE.format(site=self.base_url, context=self.context, goal=self.goal, actions=self.ACTIONS, candidate=candidate))) + len(ENCODING.encode(prompts.SYS_MAKES_SENSE))
        state, _ = OAI_GOOD.handle_response(prompts.SYS_MAKES_SENSE, prompts.USER_MAKES_SENSE.format(site=self.base_url, context=self.context, goal=self.goal, actions=self.ACTIONS, candidate=candidate))
        stripped_state = state.replace('"', '').replace('.', '').lower().strip()
        if self.verbose: print('CHECKING COMMON SENSE\n{}\n\n'.format(stripped_state))
        if stripped_state.startswith('yes') or stripped_state.endswith('yes'):
            return True, state, token_count
        return False, state, token_count

    def end_state(self):
        """
        This function prompts the user to confirm if they want to end the current state and returns a
        boolean value based on their response.
        :return: a boolean value. If the user responds with "yes" to the prompt, the function returns
        True. Otherwise, it returns False.
        """
        input_token_count = 0
        output_token_count = 0
        BM.mark('end_state')
        input_token_count = len(ENCODING.encode(prompts.USER_END_STATE.format(site=self.base_url, context=self.context, goal=self.goal, actions=self.ACTIONS))) + len(ENCODING.encode(prompts.SYS_END_STATE))
        state, _ = OAI_GOOD.handle_response(prompts.SYS_END_STATE, prompts.USER_END_STATE.format(site=self.base_url, context=self.context, goal=self.goal, actions=self.ACTIONS))
        output_token_count = len(ENCODING.encode(state))
        stripped_state = state.replace('"', '').replace('.', '').replace("'", '').lower().strip()
        if self.verbose: print('CHECKING END STATE\n{}\n\n'.format(stripped_state))
        BM.mark('end_state')
        if stripped_state.startswith('yes') or stripped_state.endswith('yes'):
            return True, state, input_token_count, output_token_count
        return False, state, input_token_count, output_token_count

    def process_screenshot(self, encoded=False):
        input_token_count = 1105
        output_token_count = 0
        BM.mark('process_screenshot')
        input_token_count += len(ENCODING.encode(prompts.USER_SCREENSHOT.format(site=self.base_url, goal=self.goal, actions=self.ACTIONS))) + len(ENCODING.encode(prompts.SYS_SCREENSHOT))
        user_prompt = handle_image(prompts.USER_SCREENSHOT.format(site=self.base_url, goal=self.goal, actions=self.ACTIONS), self.image_bytes, self.image_path, encoded=encoded)
        result, _ = OAI_IMAGE.handle_response(prompts.SYS_SCREENSHOT, user_prompt, image=True)
        output_token_count = len(ENCODING.encode(result))
        if self.verbose: print('SCREENSHOT RESPONSE\n{}\n\n'.format(result))
        BM.mark('process_screenshot')
        return result, input_token_count, output_token_count
    
    def manage_tabs(self):
        OAI.handle_response(prompts.SYS_TAB_MANAGE, prompts.USER_TAB_MANAGE.format(site=self.base_url, context=self.context, goal=self.goal, actions=self.ACTIONS, next_action=self.next_action, tabs=self.tabs))
            
    def get_page_class(self):
        BM.mark('get_page_class')
        with open('data/runtime/page_classes.json', 'r') as infile:
            data = json.load(infile)
            message, _ = OAI.handle_response(prompts.SYS_CONTEXT_CLASS, prompts.USER_CONTEXT_CLASS.format(site=self.base_url_only, page_text=self.page_text, page_classes=data))
            if self.verbose: print('PAGE CLASS\n{}\n\n'.format(message))
            for pageclass in data:
                if pageclass in message:
                    BM.mark('get_page_class')
                    return pageclass
            BM.mark('get_page_class')
            return None
            
    def retrieve_memorized_sequence(self, mode='actions'):
        BM.mark('get_cache_action')
        with open('data/runtime/cached_sequences.json', 'r') as infile:
            sequences = json.load(infile)
            if self.base_url_only not in sequences:
                BM.mark('get_cache_action')
                return None
            keys = list(sequences[self.base_url_only][mode].keys())
            formatted_items = ['({}) {}'.format(i + 1, item) for i, item in enumerate(keys)]
            keys_str = '\n'.join(formatted_items)
            message, _ = OAI.handle_response(prompts.SYS_MEMORY_RETRIEVAL, prompts.USER_MEMORY_RETRIEVAL.format(query=self.next_action, keys=keys_str))
            if self.verbose: print('MEMORY RETRIEVAL\n{}\n\n'.format(message))
            if message.lower().strip().startswith('none'):
                BM.mark('get_cache_action')
                return None
            else:
                numbers = re.findall(r'\d+', message)
                if len(numbers) > 0:
                    BM.mark('get_cache_action')
                    sequences[self.base_url_only][mode][keys[int(numbers[0]) - 1]]['timestamps'].append(datetime.today())
                    sequences[self.base_url_only][mode] = cache_timestamp_expiration(sequences[self.base_url_only][mode], max_frequency=self.cache_max_freq)
                    sequences[self.base_url_only][mode][keys[int(numbers[0]) - 1]]['frequency'] += 1
                    with open('data/runtime/cached_sequences.json', 'w') as outfile:
                        json.dump(sequences, outfile, indent=4, default=str)
                    return sequences[self.base_url_only][mode][keys[int(numbers[0]) - 1]]

    def store_memorized_sequence(self, verb, element, mode='actions'):
        BM.mark('store_cache_action')
        with open('data/runtime/cached_sequences.json', 'r') as infile:
            sequences = json.load(infile)
            if self.base_url_only in sequences:
                if mode == 'actions':
                    sequences = cache_replacement_policy(sequences, self.base_url_only, self.next_action, verb, element, mode='actions', replacement=self.cache_replacement, cache_size=self.cache_size, max_frequency=self.cache_max_freq)
                elif mode == 'goals':
                    sequences = cache_replacement_policy(sequences, self.base_url_only, self.goal, verb, element, mode='goals', replacement=self.cache_replacement, cache_size=self.cache_size, max_frequency=self.cache_max_freq)
            else:
                if self.base_url_only not in sequences:
                    sequences[self.base_url_only] = {'actions': {}, 'goals': {}}
                sequences[self.base_url_only]['actions'][self.next_action] = {'verb': verb, 'element': element, 'timestamps': [datetime.today()], 'frequency': 1}
                # sequences[self.base_url_only]['actions'][self.next_action] = {'verb': verb, 'element': element, 'prev_state': , 'next_state': ,'timestamp': datetime.today(), 'frequency': 1}
                sequences[self.base_url_only]['goals'][self.goal] = [{'verb': verb, 'element': element, 'timestamps': [datetime.today()], 'frequency': 1}]
        with open('data/runtime/cached_sequences.json', 'w') as outfile:
            json.dump(sequences, outfile, indent=4, default=str)
        BM.mark('store_cache_action')
        
    def store_state_transition(self, verb, element):
        BM.mark('store_cache_action')
        with open('data/runtime/state_transition.json', 'r') as infile:
            state_transitions = json.load(infile)
            if self.base_url_only not in state_transitions:
                state_transitions[self.base_url_only] = {}
            cur = self.STATES[self.base_url_only]
            for state in self.STATES:
                cur = cur[state]
            state_transitions[self.base_url_only][self.state] = {'html': self.html, 'goal': self.goal, 'actions': self.ACTIONS, 'next_action': self.next_action, 'verb': verb, 'element': element, 'timestamps': [datetime.today()]}

        with open('data/runtime/state_transition.json', 'w') as outfile:
            json.dump(state_transitions, outfile, indent=4)
        BM.mark('store_cache_action')
        
    def check_cache_key(self, verb, element):
        input_token_count = 0
        output_token_count = 0
        BM.mark('check_cache_key')
        input_token_count = len(ENCODING.encode(prompts.USER_MEMORY_CONFIRM.format(action=self.next_action, verb=verb, element=element))) + len(ENCODING.encode(prompts.SYS_MEMORY_CONFIRM))
        state, _ = OAI_GOOD.handle_response(prompts.SYS_MEMORY_CONFIRM, prompts.USER_MEMORY_CONFIRM.format(action=self.next_action, verb=verb, element=element))
        output_token_count = len(ENCODING.encode(state))
        stripped_state = state.replace('"', '').replace('.', '').replace("'", '').lower().strip()
        if self.verbose: print('CHECKING CACHE KEY\n{}\n\n'.format(stripped_state))
        BM.mark('check_cache_key')
        if stripped_state.startswith('yes') or stripped_state.endswith('yes'):
            return True, state, input_token_count, output_token_count
        return False, state, input_token_count, output_token_count

    def check_next_state(self, encoded=False):
        input_token_count = 1105
        output_token_count = 0
        BM.mark('process_screenshot_next_state')
        input_token_count += len(ENCODING.encode(prompts.USER_SCREENSHOT_NEXT_STATE.format(site=self.base_url, goal=self.goal, actions=self.ACTIONS))) + len(ENCODING.encode(prompts.SYS_SCREENSHOT_NEXT_STATE))
        user_prompt = handle_image(prompts.USER_SCREENSHOT_NEXT_STATE.format(site=self.base_url, goal=self.goal, actions=self.ACTIONS), self.image_bytes, self.image_path, encoded=encoded)
        result, _ = OAI_IMAGE.handle_response(prompts.SYS_SCREENSHOT_NEXT_STATE, user_prompt, image=True)
        stripped_state = result.replace('"', '').replace('.', '').replace("'", '').lower().strip()
        output_token_count = len(ENCODING.encode(result))
        if self.verbose: print('SCREENSHOT NEXT STATE\n{}\n\n'.format(result))
        BM.mark('process_screenshot_next_state')
        if stripped_state.startswith('yes') or stripped_state.endswith('yes'):
            return True, result, input_token_count, output_token_count
        return False, result, input_token_count, output_token_count

    def flatten_strip_elements(self, sublists):
        element_list = []
        for sublist in sublists:
            element_list.extend(sublist)
        stripped_list = []
        for element in element_list:
            stripped_list.append(element.strip().lower())
        return stripped_list
            
    def limit_elements(self, elements, n_tries=3):
        input_token_count = 0
        output_token_count = 0
        limited_elements = []
        cur_try = 0
        while len(limited_elements) < 1 and cur_try < n_tries:
            input_token_count += len(ENCODING.encode(prompts.USER_CSS_SELECTOR.format(next_action=self.next_action))) + len(ENCODING.encode(prompts.SYS_CSS_SELECTOR))
            message, _ = OAI.handle_response(prompts.SYS_CSS_SELECTOR, prompts.USER_CSS_SELECTOR.format(next_action=self.next_action))
            output_token_count += len(ENCODING.encode(message))
            options = message.split('\n')
            options = list(set(option.lower() for option in options))
            if self.verbose: print('LIMITING STRINGS\n{}\n\n'.format(options))
            # stripped_list = self.flatten_strip_elements(elements)
            for element in elements:
                for option in options:
                    if option in element and element not in limited_elements:
                        limited_elements.append(element)
            cur_try += 1
        return limited_elements, options, input_token_count, output_token_count

    def get_element(self, element, sublists):
        BM.mark('element_cache_matching')
        stripped_list = self.flatten_strip_elements(sublists)
        element = element.strip().lower()
        if element in stripped_list:
            return stripped_list.index(element)
        else:
            matches = get_close_matches(element, stripped_list, n=1, cutoff=0.95)
            if len(matches) > 0:
                return stripped_list.index(matches[0])
        BM.mark('element_cache_matching')
        return None
    
    def get_high_level_element(self, candidate: str):
        # token_count = len(ENCODING.encode(prompts.USER_HIGH_LEVEL_ACTION.format(site=self.base_url, context=self.context, candidate=candidate))) + len(ENCODING.encode(prompts.SYS_HIGH_LEVEL_ACTION))
        high_level_element, _ = OAI.handle_response(prompts.SYS_HIGH_LEVEL_ACTION, prompts.USER_HIGH_LEVEL_ACTION.format(site=self.base_url, context=self.context, candidate=candidate))
        return high_level_element
    
    def filter_specific_elements(self, interactables):
        flattened_interactables_map = {}
        cleaned_elements_map = {}
        all_elements = {}
        flattened_interactables = []
        elements = []
        sublists = []
        current_sublist = []
        current_tokens = 0
        counter = 0
        for tag in interactables:
            for i in range(tag.count()):
                counter += 1
        print('TOTAL ELEMENTS: {}'.format(counter))
        for tag in interactables:
            for i in range(tag.count()):
                try:
                    element = tag.nth(i).evaluate("el => el.outerHTML", timeout=500)
                    if element not in all_elements:
                        all_elements[element] = 1
                        flattened_interactables.append(tag.nth(i))
                        stripped_lowered_element = element.lower().strip()
                        if element not in flattened_interactables_map:
                            flattened_interactables_map[stripped_lowered_element] = len(flattened_interactables) - 1
                        elements.append(stripped_lowered_element)
                except Exception as e:
                    if self.verbose: print(tag, e)
                    pass
        for element in elements:
            cleansed_element = clean_element(element)
            if cleansed_element is not None and len(cleansed_element) < 4000:
                cleaned_elements_map[cleansed_element] = element
                encoded_element = ENCODING.encode(cleansed_element)
                if current_tokens + len(encoded_element) <= self.MAX_TOKENS:
                    current_sublist.append(cleansed_element)
                    current_tokens += len(encoded_element)
                else:
                    sublists.append(current_sublist)
                    current_sublist = [cleansed_element]
                    current_tokens = len(encoded_element)
        if current_sublist:
            sublists.append(current_sublist)
        return sublists, flattened_interactables, flattened_interactables_map, cleaned_elements_map
    
    def filter_elements(self, limit_elements=True):
        BM.mark('filter_elements')
        flattened_interactables_map = {}
        cleaned_elements_map = {}
        all_elements = {}
        flattened_interactables = []
        elements = []
        sublists = []
        current_sublist = []
        current_tokens = 0
        counter = 0
        for tag in self.interactables:
            for i in range(tag.count()):
                counter += 1
        print('TOTAL ELEMENTS: {}'.format(counter))
        for tag in self.interactables:
            for i in range(tag.count()):
                try:
                    element = tag.nth(i).evaluate("el => el.outerHTML", timeout=500)
                    if element not in all_elements:
                        all_elements[element] = 1
                        flattened_interactables.append(tag.nth(i))
                        stripped_lowered_element = element.lower().strip()
                        if element not in flattened_interactables_map:
                            flattened_interactables_map[stripped_lowered_element] = len(flattened_interactables) - 1
                        elements.append(stripped_lowered_element)
                except Exception as e:
                    if self.verbose: print(tag, e)
                    pass

        if limit_elements:
            BM.mark('limit_elements')
            elements, _, _, _ = self.limit_elements(elements)
            BM.mark('limit_elements')
        print('LIMITED ELEMENTS: {}'.format(len(elements)))
        for element in elements:
            cleansed_element = clean_element(element)
            if cleansed_element is not None and len(cleansed_element) < 4000:
                cleaned_elements_map[cleansed_element] = element
                encoded_element = ENCODING.encode(cleansed_element)
                if current_tokens + len(encoded_element) <= self.MAX_TOKENS:
                    current_sublist.append(cleansed_element)
                    current_tokens += len(encoded_element)
                else:
                    sublists.append(current_sublist)
                    current_sublist = [cleansed_element]
                    current_tokens = len(encoded_element)
        if current_sublist:
            sublists.append(current_sublist)

        BM.mark('filter_elements')
        with open('data/examples/all_elements.html', 'w') as outfile:
            counter = 1
            for element, _ in all_elements.items():
                outfile.write('({}) {}\n\n'.format(counter, element))
                counter += 1
        with open('data/examples/elements.html', 'w') as outfile:
            for sublist in sublists:
                for i, element in enumerate(sublist):
                    outfile.write('({}) {}\n\n'.format(i, element))
        with open('data/examples/limited_elements.html', 'w') as outfile:
            for element in elements:
                outfile.write('{}\n\n'.format(element))

        return sublists, flattened_interactables, flattened_interactables_map, cleaned_elements_map
    
    def run(self):
        n_tries = 3
        limit_elements = True
        sequence = False
        # sequence = self.retrieve_memorized_sequence(mode='goals')
        if sequence:
            print('IMPLEMENT')
            exit()
        self.page_text = ParseHtml(self.html).get_page_text()
        encoded_element = ENCODING.encode(self.page_text)
        if len(encoded_element) > 500:
            self.page_text = ENCODING.decode(encoded_element[:500])
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks to be executed in parallel
            future_to_function = {
                executor.submit(self.process_screenshot): 'process_screenshot',
                executor.submit(self.get_page_context): 'get_page_context',
                # executor.submit(self.get_page_class): 'get_page_class',
            }
        results = {}
        for future in as_completed(future_to_function):
            func_name = future_to_function[future]
            try:
                result = future.result()
                results[func_name] = result
            except Exception as exc:
                print(f'{func_name} generated an exception: {exc}')
        print(results)
        
        self.context, _, _ = results['get_page_context']
        if len(self.ACTIONS) > 0 and self.end_state()[0]:
            if self.verbose: print('SUCCESS END STATE')
            # self.store_memorized_sequence(verb, newelement, mode='goals')
            return True
        self.next_action, _, _ = results['process_screenshot']
        # self.page_class = results['get_page_class']

        sequence = self.retrieve_memorized_sequence()
        high_level_element = ' '.join(self.next_action.split()[1:])
        if sequence:
            verb = sequence['verb']
            newelement = sequence['element']
            print(newelement)
            action = ACTION_VERB_MAP[verb]
            tags = get_interactable(self.tab, newelement)
            if tags[0].count() > 0:
                sublists, flattened_interactables, flattened_interactables_map, cleaned_elements_map = self.filter_specific_elements(tags)
            else:
                sublists, flattened_interactables, flattened_interactables_map, cleaned_elements_map = self.filter_elements(limit_elements=limit_elements)
            tag_id = self.get_element(newelement, sublists)
            if self.verbose: print('CACHE MATCH FOUND {}\n{} {}\n\n'.format(tag_id, verb, newelement))
        else:
            sublists, flattened_interactables, flattened_interactables_map, cleaned_elements_map = self.filter_elements(limit_elements=limit_elements)
            if self.verbose: print('\n========================================\n')
            if self.verbose: print('STATE:\n------\nWebsite visited: {site}\nPage context: {context}\nUser goal: {goal}\nPrior actions: {actions}\n\n'.format(site=self.base_url, context=self.context, goal=self.goal, actions=self.ACTIONS))
            tag_id, _, newelement, action, verb, message, _ = self.next_element(sublists, n_tries=n_tries)
            if tag_id is None:
                if self.verbose: print('ERROR')
            print(newelement)
        if newelement in cleaned_elements_map:
            newtag = flattened_interactables[flattened_interactables_map[cleaned_elements_map[newelement]]]
        else:
            try:
                matched_element = get_close_matches(newelement, list(cleaned_elements_map.keys()), n=1, cutoff=0.95)[0]
                newtag = flattened_interactables[flattened_interactables_map[matched_element]]
            except Exception as e:
                newtag = None            
        print(newtag)
        bounds = newtag.bounding_box()

        candidate = '{} {}'.format(verb, newelement)
        high_level_action = '{} {}'.format(verb, high_level_element)
        if self.verbose: print(candidate)
        if self.verbose: print('HIGH LEVEL ACTION\n{}\n\n'.format(high_level_action))

        text_field_message, text_field, text_field_censored, _, _ = self.text_fields(high_level_action, verb)
        if text_field:
            high_level_action = '{} {} in {}'.format(verb, text_field_censored, high_level_element)
        if self.verbose: print('TEXT FIELD INPUT\n{}\n\n'.format(text_field_censored))
        if self.verbose: print('========================================\n')
        
        self.newtag = newtag
        self.newelement = newelement
        self.action = action
        self.text_field = text_field
        self.bounds = bounds
        
        if not sequence:
            if self.check_cache_key(verb, newelement)[0]:
                self.store_memorized_sequence(verb, newelement)
            # with ThreadPoolExecutor(max_workers=3) as executor:
            #     # Submit tasks to be executed in parallel
            #     future_to_function = {
            #         executor.submit(self.makes_sense, verb, newelement, text_field_censored): 'makes_sense',
            #     }
            # for future in as_completed(future_to_function):
            #     func_name = future_to_function[future]
            #     try:
            #         result = future.result()
            #         results[func_name] = result
            #     except Exception as exc:
            #         print(f'{func_name} generated an exception: {exc}')
            
            # if results['makes_sense'][0]:
            #     self.store_memorized_sequence(verb, newelement)
            # else:
            #     if self.verbose: print('DOESNT MAKE SENSE')
        self.ACTIONS.append(high_level_action)
        return False


def recurse_interactables(actor, context, page: Page, depth, depth_limit=10, action_limit=1000):
    print(page.url)
    root = os.path.join('data', 'runtime')
    prev_url = page.url.replace('/', '-|slash|-')
    if depth > depth_limit or PATH_NUM > action_limit:
        return
        # if tag.get_attribute('target') == '_blank' and action == click:
        #     html, interactables, traffic, page = interact_new_page(context, tag, action, PATH_NUM)
        # else:
    BM.mark('perform_action')
    html, interactables, traffic = interact(actor.newtag, page, actor.action, PATH_NUM, actor.text_field)
    if '_blank' in actor.newelement:
        html, interactables, traffic, page, opened_new_tab = interact_new_page(context, actor.newtag, page, actor.action, PATH_NUM, actor.text_field)
    url = page.url.replace('/', '-|slash|-')
    BM.mark('perform_action')
    BM.mark('take_screenshot')
    image_path, image_bytes = screenshot(page)
    BM.mark('take_screenshot')
    BM.mark('update_system')
    actor.update(base_url=page.url, interactables=interactables, html=html, image_path=image_path, image_bytes=image_bytes, tab=page, tabs=context.pages)
    BM.mark('update_system')
    if actor.run():
        return
    storage = get_storage(context)
    # with check_and_open_file(os.path.join(root, 'network-traffic'), url, ext='json', PATH_NUM=PATH_NUM) as outfile:
    #     json.dump(traffic, outfile)
    # with check_and_open_file(os.path.join(root, 'html'), url, ext='html', PATH_NUM=PATH_NUM) as outfile:
    #     outfile.write(html)
    # with check_and_open_file(os.path.join(root, 'interactions'), prev_url, ext='csv', mode='a') as outfile:
    #     if actor.action == type_text or actor.action == drag or actor.action == type_submit or actor.action == check or actor.action == select or actor.action == exploration or actor.action == upload_file:
    #         outfile.write('{}-|split|-,{}-|split|-,{}-|split|-,{}-|split|-,{}-|split|-,{}\n'.format(PATH_NUM, actor.action, actor.bounds, url, actor.newelement.replace('\n', '').replace('  ', ''), actor.text_field))
    #     else:
    #         outfile.write('{}-|split|-,{}-|split|-,{}-|split|-,{}-|split|-,{}\n'.format(PATH_NUM, actor.action, actor.bounds, url, actor.newelement.replace('\n', '').replace('  ', '')))
    # with check_and_open_file(os.path.join(root, 'storage'), url, ext='json', PATH_NUM=PATH_NUM) as outfile:
    #     json.dump(storage, outfile)
    if actor.newtag is None:
        return
    recurse_interactables(actor, context, page, depth + 1, depth_limit=depth_limit, action_limit=action_limit)


def test_playwright(playwright, args):
    global PATH_NUM
    global TOTAL
    global PROGRESS
    global EXCEPTIONS
    global OAI
    # global OAI_LONG
    global OAI_GOOD
    global ACTIONS
    global ELEMENTS
    global ENCODING
    root = os.path.join('data', 'runtime')
    # chromium = playwright.chromium
    # browser = chromium.launch(headless=False)
    firefox = playwright.firefox
    browser = firefox.launch(headless=False)
    # OAI = OpenAIAPI(model='gpt-4')
    # OAI = OpenAIAPI()
    # OAI = OpenAIAPI(model=args.model)
    # OAI_LONG = OpenAIAPI(model=args.model_long)
    # OAI_GOOD = OpenAIAPI(model=args.model_good)
    # ENCODING = tiktoken.encoding_for_model(args.model_long)
    
    if args.task and args.url:
        ACTIONS = []
        ELEMENTS = []
        if args.auth:
            context = browser.new_context(
                storage_state='storage_combined.json',
                # record_video_dir="data/runtime/videos/",
                # record_video_size={"width": 640, "height": 480}
            )
        else:
            context = browser.new_context(
                # record_video_dir="data/runtime/videos/",
                # record_video_size={"width": 640, "height": 480}
            )
        context.set_default_timeout(20000)
        page = context.new_page()
        base_url = args.url.replace('https://', '').replace('http://', '')
        if args.verbose: print('\n========================================\n')
        if args.verbose: print('WEBSITE\n{}'.format(base_url))
        PATH_NUM = 0
        TOTAL = 0
        EXCEPTIONS = 0
        # PROGRESS = tqdm(bar_format="Interactions: {postfix[0]} / {total} | Elapsed: {elapsed} | Exceptions: {postfix[1]}", total=TOTAL, postfix=[PATH_NUM, EXCEPTIONS])
        BM.mark('visit_url')
        html, interactables, traffic = exploration(context, page, base_url)
        BM.mark('visit_url')
        url = page.url.replace('/', '-|slash|-')
        BM.mark('take_screenshot')
        image_path, image_bytes = screenshot(page)
        BM.mark('take_screenshot')
        BM.mark('initialize_system')
        actor = PageStateActor(page.url, base_url, args.task, interactables, html, page, context.pages, args.max_tokens, image_path, image_bytes, args.verbose)
        BM.mark('initialize_system')
        finished = actor.run()
        storage = get_storage(context)
        # page.video.save_as(os.path.join('data', 'runtime', 'videos', '{}.webm'.format(url)))
        # with check_and_open_file(os.path.join(root, 'network-traffic'), url, ext='json') as outfile:
        #     json.dump(traffic, outfile)
        # with check_and_open_file(os.path.join(root, 'html'), url, ext='html') as outfile:
        #     outfile.write(html)
        # with check_and_open_file(os.path.join(root, 'storage'), url, ext='json') as outfile:
        #     json.dump(storage, outfile)
        if not finished:
            recurse_interactables(actor, context, page, 0, depth_limit=100, action_limit=1500)
        context.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0, help='Index for Tranco list')
    parser.add_argument('--url', type=str, default='', help='')
    parser.add_argument('--task', type=str, default='', help='')
    parser.add_argument('--auth', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=12000, help='')
    parser.add_argument('--model', type=str, choices=['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106'], default='gpt-3.5-turbo', help='')
    parser.add_argument('--model-long', type=str, choices=['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-16k', 'gpt-4'], default='gpt-3.5-turbo-16k', help='')
    parser.add_argument('--model-good', type=str, choices=['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-16k', 'gpt-4'], default='gpt-4', help='')
    parser.add_argument('--verbose', action='store_true', help='')
    args = parser.parse_args()

    with sync_playwright() as playwright:
        test_playwright(playwright, args)
