import nopecha
from sensitive.objects import nopecha_api_key


nopecha.api_key = nopecha_api_key


# recaptcha, hcaptcha, textcaptcha, funcaptcha, awscaptcha
def solve_captcha(captcha_type='recaptcha', task=None, image_data=None, audio_data=None, urls=None, grid=None):
    # print(captcha_type, task, image_data, audio_data, urls, grid)
    if captcha_type == 'recaptcha' and urls:
        params = {'type': captcha_type, 'task': task, 'image_urls': urls, 'grid': grid}
    elif captcha_type == 'recaptcha' and image_data:
        params = {'type': captcha_type, 'task': task, 'image_data': image_data, 'grid': grid}
    elif captcha_type == 'hcaptcha':
        params = {'type': captcha_type, 'task': task, 'image_urls': urls}
    elif captcha_type == 'funcaptcha':
        params = {'type': captcha_type, 'task': task, 'image_data': image_data}
    elif captcha_type == 'textcaptcha':
        params = {'type': captcha_type, 'task': task, 'image_data': image_data}
    response = nopecha.Recognition.solve(**params)
    print(response)
    return response


def detect_captcha():
    frame_locator = page.frame_locator('[title*="{}"]'.format('reCAPTCHA'))
    frame_locator = page.frame_locator('[title*="{}"]'.format('Widget containing checkbox for hCaptcha'))
    frame_locator_locator = frame_locator.frame_locator('[id="{}"]'.format('CaptchaFrame'))
    frame_locator = page.frame_locator('[id*="{}"]'.format('fc-iframe-wrap'))
    print(frame_locator)
    frame_locator_locator = frame_locator.frame_locator('[id="{}"]'.format('CaptchaFrame'))


def detect_cloudflare(page):
    html = page.content()
    parser = ParseHtml(html=html)
    text = parser.get_page_text().lower()
    if 'checking your browser' in text:
        return True
    else:
        return False