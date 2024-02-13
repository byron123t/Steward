import prompts
from API import OpenAIAPI, handle_image
from difflib import get_close_matches


OAI = OpenAIAPI(model='gpt-3.5-turbo-1106')
OAI_GOOD = OpenAIAPI(model='gpt-4-1106-preview')
message, _ = OAI.handle_response(prompts.SYS_CSS_SELECTOR, prompts.USER_CSS_SELECTOR.format(site='https://www.united.com/en/us', context='Airline ticket booking and travel information page for United Airlines.', next_action='click "Check-in" tab'))
print(message)
options = message.split('\n')
options = list(set(option.lower() for option in options))
yes_elements = []
with open('elements.html', 'r') as infile:
    elements = []
    for line in infile:
        line = line.strip().lower()
        if len(line) > 0:
            elements.append(line)
    for element in elements:
        for option in options:
            if option in element:
                yes_elements.append(element)
            
print(yes_elements)
