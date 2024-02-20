# click, click + drag, text input,
# touch, pinch, swipe, 
# Maybe can convert natural language commands to Playwright code generation
from ast import Str
from pickle import NEWTRUE
import os
from difflib import get_close_matches
from playwright.sync_api import Page, expect, sync_playwright, Locator


class Network:
    
    def __init__(self, page: Page):
        self.network_traffic = []
        page.on('request', self.handle_request)
        page.on('response', self.handle_response)

    def handle_request(self, request):
        data_dict = request.all_headers()
        data_dict['type'] = 'request'
        data_dict['method'] = request.method
        data_dict['url'] = request.url
        try:
            data_dict['data'] = request.post_data
        except UnicodeDecodeError as e:
            data_dict['data'] = None
        data_dict['status'] = None
        self.network_traffic.append(data_dict)

    def handle_response(self, response):
        data_dict = response.all_headers()
        data_dict['type'] = 'response'
        data_dict['method'] = None
        data_dict['url'] = response.url
        data_dict['data'] = response.header_values('set-cookie')
        data_dict['status'] = response.status
        self.network_traffic.append(data_dict)

    def get_traffic(self):
        return self.network_traffic
    

def check_file_length(path, filename):
    max_len = os.statvfs(path).f_namemax
    ext = filename.split('.')[-1]
    if len(filename) > max_len:
        return filename[:max_len - 10] + ext
    return filename


def check_and_open_file(path, url, ext='csv', mode='w', PATH_NUM=None):
    if PATH_NUM:
        filename = '{}{}.{}'.format(url.replace('/', '-|slash|-'), PATH_NUM, ext)
    else:
        filename = '{}.{}'.format(url.replace('/', '-|slash|-'), ext)
    filename = check_file_length(path, filename)
    return open(os.path.join(path, filename), mode)


def type_text(interactable: Locator=None, text: Str=None, delay: float=0.147):
    interactable.type(text, delay=delay)


def type_submit(interactable: Locator=None, text: Str=None, delay: float=0.147):
    interactable.type(text, delay=delay)
    interactable.press('Enter')


def enter(interactable: Locator=None, key: Str=None):
    interactable.press('Enter')


def type_key(interactable: Locator=None, key: Str=None):
    interactable.press(key)


def hover(interactable: Locator=None):
    interactable.hover()


def drag(src_interactable: Locator=None, tar_interactable: Locator=None):
    src_interactable.drag_to(tar_interactable)


def click(page: Page=None, interactable: Locator=None, double=False, button='left', modifiers=[], position=None, navigation=False):
    if double:
        if navigation:
            with page.expect_navigation():
                interactable.dblclick()
        else:
            interactable.dblclick()
    else:
        if position is not None:
            if navigation:
                with page.expect_navigation():
                    interactable.click(button=button, modifiers=modifiers, position=position, force=True)
            else:
                interactable.click(button=button, modifiers=modifiers, position=position, force=True)
        else:
            if navigation:
                with page.expect_navigation():
                    interactable.click(button=button, modifiers=modifiers, force=True)
            else:
                interactable.click(button=button, modifiers=modifiers, force=True)


def right_click(page: Page=None, interactable: Locator=None, double=False, button='right', modifiers=[], position=None, navigation=False):
    if double:
        if navigation:
            with page.expect_navigation():
                interactable.dblclick()
        else:
            interactable.dblclick()
    else:
        if position is not None:
            if navigation:
                with page.expect_navigation():
                    interactable.click(button=button, modifiers=modifiers, position=position, force=True)
            else:
                interactable.click(button=button, modifiers=modifiers, position=position, force=True)
        else:
            if navigation:
                with page.expect_navigation():
                    interactable.click(button=button, modifiers=modifiers, force=True)
            else:
                interactable.click(button=button, modifiers=modifiers, force=True)


def focus(interactable: Locator=None):
    interactable.focus()


def check(interactable: Locator=None):
    interactable.check()
    
def select(interactable: Locator=None, options=None):
    interactable.select_option(options)


def upload_file(page: Page=None, interactable: Locator=None, file=None):
    page.on('filechooser', lambda file_chooser: file_chooser.set_files(file))
    interactable.set_input_files(file)


def screenshot(page: Page, PATH_NUM=None, save=True):
    path = 'data/screenshots/'
    if PATH_NUM:
        filename = '{}{}.jpeg'.format(page.url.replace('/', '-|slash|-'), PATH_NUM)
    else:
        filename = '{}.jpeg'.format(page.url.replace('/', '-|slash|-'))
    filename = check_file_length(path, filename)
    if save:
        screenshot_bytes = page.screenshot(path=os.path.join(path, filename))
    else:
        screenshot_bytes = page.screenshot()
    return os.path.join(path, filename), screenshot_bytes


def get_storage(context):
    return context.storage_state()


def modify_request(route):
    headers = route.request.headers
    del headers['cookies']
    route.continue_(headers=headers)


def modify_response(page: Page, route):
    response = page.request.fetch(route.request)
    body = response.text()
    body = body.replace("<title>", "<title>My prefix:")
    route.fulfill(
        # Pass all fields from the response.
        response=response,
        # Override response body.
        body=body,
        # Force content type to be html.
        headers={**response.headers, "content-type": "text/html"},
    )


def exploration(context, page: Page, url):
    network = Network(page)
    if url.startswith('https://') or url.startswith('http://'):
        pass
    elif not url.startswith('https://'):
        url = 'https://{}'.format(url)
    elif not url.startswith('http://'):
        url = 'http://{}'.format(url)
    try:
        page.goto(url, wait_until='networkidle')
    except Exception as e:
        page.goto(url)
    try:
        page.wait_for_load_state('networkidle', timeout=2000)
    except Exception as e:
        pass
    try:
        page.wait_for_load_state('domcontentloaded', timeout=2000)
    except Exception as e:
        pass
    # screenshot(page)
    return get_html(page), get_all_interactables(page), network.get_traffic()


def interact(tag, page: Page, action, PATH_NUM, param2=None):
    network = Network(page)
    try:
        if action == type_text or action == drag or action == type_submit or action == check or action == select or action == exploration or action == upload_file:
            action(tag, param2)
        elif action == click:
            action(page, tag)
        else:
            action(tag)
    except Exception as e:
        if action == type_text or action == drag or action == type_submit or action == check or action == select or action == exploration or action == upload_file:
            action(tag, param2)
        elif action == click:
            action(page, tag)
        else:
            action(tag)
    try:
        page.wait_for_load_state('networkidle', timeout=5000)
    except Exception as e:
        pass
    try:
        page.wait_for_load_state('domcontentloaded', timeout=5000)
    except Exception as e:
        pass
    # screenshot(page, PATH_NUM)
    return get_html(page), get_all_interactables(page), network.get_traffic()


def interact_new_page(context, tag, page: Page, action, PATH_NUM, param2=None):
    try:
        with context.expect_page() as new_page_info:
            if action == type_text or action == drag or action == type_submit or action == check or action == select or action == exploration or action == upload_file:
                action(tag, param2)
            elif action == click:
                action(page, tag)
            else:
                action(tag)
        new_page = new_page_info.value
        network = Network(new_page)
    except Exception as e:
        with context.expect_page() as new_page_info:
            if action == type_text or action == drag or action == type_submit or action == check or action == select or action == exploration or action == upload_file:
                action(tag, param2)
            elif action == click:
                action(page, tag)
            else:
                action(tag)
        new_page = new_page_info.value
        network = Network(new_page)
    try:
        new_page.wait_for_load_state('networkidle')
    except Exception as e:
        new_page.wait_for_load_state('networkidle')
    try:
        new_page.wait_for_load_state('domcontentloaded')
    except Exception as e:
        new_page.wait_for_load_state('domcontentloaded')
    # screenshot(new_page, PATH_NUM)
    return get_html(new_page), get_all_interactables(new_page), network.get_traffic(), new_page


def get_html(page: Page):
    return page.content()


def get_all_interactables(page: Page):
    interactables = []
    interactables.append(page.locator('button:visible'))
    interactables.append(page.locator('a:visible'))
    interactables.append(page.locator('input:visible'))
    interactables.append(page.locator('select:visible'))
    interactables.append(page.locator('textarea:visible'))
    interactables.append(page.locator('[role*="{}"]:visible'.format('radio')))
    interactables.append(page.locator('[role*="{}"]:visible'.format('option')))
    interactables.append(page.locator('[role*="{}"]:visible'.format('checkbox')))
    interactables.append(page.locator('[role*="{}"]:visible'.format('button')))
    interactables.append(page.locator('[role*="{}"]:visible'.format('tab')))
    interactables.append(page.locator('[role*="{}"]:visible'.format('textbox')))
    interactables.append(page.locator('[role*="{}"]:visible'.format('link')))
    interactables.append(page.locator('[role*="{}"]:visible'.format('menuitem')))
    interactables.append(page.locator('[role*="{}"]:visible'.format('tabpanel')))
    interactables.append(page.locator('[onclick]:visible'))
    interactables.append(page.locator('[href]:visible'))
    interactables.append(page.locator('[aria-controls]:visible'))
    interactables.append(page.locator('[aria-label]:visible'))
    
    return interactables


def get_all_clickables(page: Page):
    interactables = []
    interactables.append(page.locator('button'))
    interactables.append(page.locator('a'))
    interactables.append(page.locator('select'))
    interactables.append(page.locator('[class*="{}"]'.format('ui-accordion')))
    interactables.append(page.locator('[class*="{}"]'.format('ui-draggable-handle')))
    interactables.append(page.locator('[class*="{}"]'.format('ui-droppable')))
    interactables.append(page.locator('[class*="{}"]'.format('ui-sortable-handle')))
    interactables.append(page.locator('[class*="{}"]'.format('ui-accordion-header-collapsed')))
    interactables.append(page.locator('[class*="{}"]'.format('ui-datepicker-prev')))
    interactables.append(page.locator('[class*="{}"]'.format('ui-datepicker-next')))
    interactables.append(page.locator('[class*="{}"]'.format('ui-slider')))
    interactables.append(page.locator('[class*="{}"]'.format('tab')))
    interactables.append(page.locator('[class*="{}"]'.format('suggestion')))
    interactables.append(page.locator('[class*="{}"]'.format('checkbox')))
    interactables.append(page.locator('[class*="{}"]'.format('button')))
    interactables.append(page.locator('[class*="{}"]'.format('radio')))
    interactables.append(page.locator('[class*="{}"]'.format('option')))
    interactables.append(page.locator('[class*="{}"]'.format('link')))
    interactables.append(page.locator('[role*="{}"]'.format('radio')))
    interactables.append(page.locator('[role*="{}"]'.format('option')))
    interactables.append(page.locator('[role*="{}"]'.format('checkbox')))
    interactables.append(page.locator('[role*="{}"]'.format('button')))
    interactables.append(page.locator('[role*="{}"]'.format('tab')))
    interactables.append(page.locator('[role*="{}"]'.format('link')))
    interactables.append(page.locator('[role*="{}"]'.format('menuitem')))
    interactables.append(page.locator('[data-event*=""]'))
    interactables.append(page.locator('[data-action*=""]'))
    return interactables


def get_all_typeables(page: Page):
    interactables = []
    interactables.append(page.locator('input'))
    interactables.append(page.locator('textarea'))
    interactables.append(page.locator('[class*="{}"]'.format('input')))
    interactables.append(page.locator('[class*="{}"]'.format('textbox')))
    interactables.append(page.locator('[role*="{}"]'.format('textbox')))
    interactables.append(page.locator('[data-event*=""]'))
    interactables.append(page.locator('[data-action*=""]'))
    return interactables


def get_interactable_with_text(page: Page, string):
    interactables = []
    interactables.append(page.locator('button:has-text("{}")'.format(string)))
    interactables.append(page.locator('a:has-text("{}")'.format(string)))
    interactables.append(page.locator('input:has-text("{}")'.format(string)))
    interactables.append(page.locator('select:has-text("{}")'.format(string)))
    return interactables


def get_interactable_with_id(page: Page, string):
    interactables = []
    interactables.append(page.locator('button[id*="{}"]'.format(string)))
    interactables.append(page.locator('a[id*="{}"]'.format(string)))
    interactables.append(page.locator('input[id*="{}"]'.format(string)))
    interactables.append(page.locator('select[id*="{}"]'.format(string)))
    return interactables


def get_interactable_with_class(page: Page, string):
    interactables = []
    interactables.append(page.locator('button[class*="{}"]'.format(string)))
    interactables.append(page.locator('a[class*="{}"]'.format(string)))
    interactables.append(page.locator('input[class*="{}"]'.format(string)))
    interactables.append(page.locator('select[class*="{}"]'.format(string)))
    return interactables
