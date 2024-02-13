from sensitive import openai_key, azure_key, azure_url
import openai, os, json, base64


def encode_image(image_path:str):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def handle_image(user_prompt:str, image_bytes:str=None, image_path:str=None):
    if image_bytes:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        new_user_prompt = [{'type': 'text', 'text': user_prompt},{'type': 'image_url','image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}}]
        return new_user_prompt
    elif image_path:
        base64_image = encode_image(image_path)
        new_user_prompt = [{'type': 'text', 'text': user_prompt},{'type': 'image_url','image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}}]
        return new_user_prompt


class OpenAIAPI:
    def __init__(self, model:str='gpt-3.5-turbo', mode:str='openai', max_tries:int=5, temperature:float=1, presence_penalty:float=0, frequency_penalty:float=0, verbose:bool=False):
        self.mode = mode
        self.model = model
        self.verbose = verbose
        self.max_tries = max_tries
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        if self.model == 'gpt-3.5-turbo':
            self.deployment = 'turbo'
        elif self.model == 'gpt-3.5-turbo-16k':
            self.deployment = 'long'
        self.init_keys()
        
    def init_keys(self):
        if self.mode == 'openai':
            openai.api_key = openai_key
            openai.api_type = 'openai'
            openai.api_version = None
            openai.api_base = 'https://api.openai.com/v1'
        elif self.mode == 'azure':
            openai.api_key = azure_key
            openai.api_type = 'azure'
            openai.api_version = '2023-05-15'
            openai.api_base = azure_url

    def handle_response(self, sys_prompt:str=None, user_prompt:str=None, chat_history:list=None, keyword:str=None, include_role:bool=False, stream:bool=False, name:str=None, image:bool=False):
        self.init_keys()
        for _ in range(0, self.max_tries):
            try:
                if chat_history:
                    chat = chat_history
                elif sys_prompt and user_prompt:
                    chat = [{'role': 'system', 'content': sys_prompt},{'role': 'user', 'content': user_prompt}]
                else:
                    raise ValueError('Either chat_history or sys_prompt and user_prompt must be provided.')
                if self.mode == 'openai':
                    if image:
                            response = openai.ChatCompletion.create(
                            model=self.model,
                            messages=chat,
                            stream=stream,
                            temperature=self.temperature,
                            presence_penalty=self.presence_penalty,
                            frequency_penalty=self.frequency_penalty,
                            max_tokens=500)
                    else:
                        response = openai.ChatCompletion.create(
                            model=self.model,
                            messages=chat,
                            stream=stream,
                            temperature=self.temperature,
                            presence_penalty=self.presence_penalty,
                            frequency_penalty=self.frequency_penalty)
                elif self.mode == 'azure':
                    response = openai.ChatCompletion.create(
                        engine=self.deployment,
                        messages=chat,
                        stream=stream,
                        temperature=self.temperature,
                        presence_penalty=self.presence_penalty,
                        frequency_penalty=self.frequency_penalty)
                if stream:
                    return response, None
                else:
                    if 'content' not in response['choices'][0]['message']:
                        return None, None
                    if 'finish_reason' in response['choices'][0] and response['choices'][0]['finish_reason'] == 'stop' or 'finish_details' in response['choices'][0] and response['choices'][0]['finish_details']['type'] == 'stop':
                        message = response['choices'][0]['message']['content']
                    else:
                        if self.verbose: print(response)
                        message = response['choices'][0]['message']['content']
                    if len(response['choices']) > 0:
                        if self.verbose: print(response['choices'])
                        if keyword:
                            for item in response['choices']:
                                if item['finish_reason'] == 'stop':
                                    if keyword in item['message']['content']:
                                        message = item['message']['content']
                    if include_role:
                        message = {'content': message, 'role': 'assistant'}
                    if name:
                        with open('data/train_test/{}/{}.txt'.format(name, len(os.listdir('data/train_test/{}'.format(name)) / 2)), 'w') as outfile:
                            json.dump({'model': self.model, 'temperature': self.temperature, 'presence_penalty': self.presence_penalty, 'frequency_penalty': self.frequency_penalty, 'response': message, 'system_prompt': sys_prompt, 'user_prompt': user_prompt}, outfile, indent=4)
                    return message, response
            except openai.error.APIConnectionError as e:
                print(e)
                continue
            except openai.error.RateLimitError as e:
                print(e)
                continue
            except openai.error.ServiceUnavailableError as e:
                print(e)
                continue
            except openai.error.InvalidRequestError as e:
                print(e)
                raise Exception('Invalid request content restricted')
        raise Exception('Max tries exceeded')
