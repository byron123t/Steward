from action import *
import playwright
from playwright.sync_api import Page, expect
from parse_html import ParseHtml
from time import sleep


def process_interactables(interactables):
    flattened_interactables = []
    counter = 0
    for tag in interactables:
        for i in range(tag.count()):
            flattened_interactables.append(tag.nth(i))
            print(counter)
            print(tag.nth(i).evaluate("el => el.outerHTML", timeout=500))
            print()
            print()
            counter += 1
    return flattened_interactables


def test_playwright(playwright):
    global PATH_NUM
    global TOTAL
    global PROGRESS
    global EXCEPTIONS
    global OAI
    global OAI_GOOD
    global ACTIONS
    global ELEMENTS
    global ENCODING
    root = os.path.join('data', 'runtime')
    firefox = playwright.firefox
    browser = firefox.launch(headless=False)
    context = browser.new_context(
        # record_video_dir="data/runtime/videos/",
        # record_video_size={"width": 640, "height": 480}
    )
    context.set_default_timeout(6000)
    page = context.new_page()
    print(context.pages)
    html, interactables, traffic = exploration(context, page, 'https://rtcl.eecs.umich.edu/rtclweb/people/')
    flattened_interactables = process_interactables(interactables)
    print(context.pages)
    html, interactables, traffic = interact(flattened_interactables[15], page, click, PATH_NUM=None)
    flattened_interactables = process_interactables(interactables)
    print(context.pages)
    html, interactables, traffic, new_page = interact_new_page(context, flattened_interactables[6], page, click, PATH_NUM=None)
    new_flattened_interactables = process_interactables(interactables)
    print(context.pages)
    html, interactables, traffic = interact(flattened_interactables[4], page, click, PATH_NUM=None)
    html, interactables, traffic = interact(new_flattened_interactables[10], new_page, click, PATH_NUM=None)
    # newnewpage = context.new_page()
    with context.expect_page() as new:
        page.evaluate("window.open('https://www.google.com')")
        page2 = new.value
    page.bring_to_front()
    sleep(2)
    print(context.pages)
    page2.bring_to_front()
    sleep(5)
    
with sync_playwright() as playwright:
    test_playwright(playwright)
