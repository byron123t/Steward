from lxml import etree
from lxml.html import clean
from zxcvbn import zxcvbn
import nltk
nltk.download('words')
from nltk.corpus import words
import math, re, statistics
from benchmark import BM



SAFE_ATTRS = frozenset({'class', 'type', 'datetime', 'readonly', 'src', 'id', 'value', 'label', 'title', 'alt', 'href', 'checked', 'selected', 'nohref', 'name', 'role', 'aria-label', 'placeholder', 'data-placeholder', 'data-action', 'data-event', 'hidden', 'aria-hidden', 'disabled', 'aria-disabled', 'aria-checked'})
CLEANER = clean.Cleaner(safe_attrs_only=True, safe_attrs=SAFE_ATTRS, forms=False, frames=False)
WORDS = set(filter(lambda word: len(word) >= 3, words.words()))
STRINGS = {}

# def is_random_string(string, threshold=3.0):
#     """Checks if a string is random based on its entropy."""
#     if len(string) > 2:
#         if string in STRINGS:
#             score = STRINGS[string]
#         else:
#             guesses = zxcvbn(string)['guesses']
#             score = math.log(float(guesses), 2) / len(string)
#             STRINGS[string] = score
#         if (score > threshold):
#             return True
#     return False

def is_random_string(string, threshold=3.0):
    """Checks if a string is random based on its entropy."""
    if len(string) > 2:
        if string in STRINGS:
            return STRINGS[string]
        else:
            guesses = zxcvbn(string)['guesses']
            score = math.log(float(guesses), 2) / len(string)
            words_in_text = re.findall(r'[A-Za-z][a-z]*|[A-Z][a-z]*', string)
            words_in_text.extend(re.split('[-, _.]', string.lower()))
            has_word = False
            for word in words_in_text:
                if word in WORDS:
                    has_word = True
            if (not has_word and score > threshold) or (not has_word and abs(math.log(float(guesses), 10) - len(string)) <= len(string)/30):
                STRINGS[string] = True
                return True
            STRINGS[string] = False
        return False
    return False


def process_node_class(node):
    # Process children
    for child in node:
        process_node_class(child)
    # If current node has any attribute
    if node.attrib:
        for attr, value in node.attrib.items():
            if attr == 'class':
                classes = node.attrib['class'].split(' ')
                # Check each class for randomness
                new_classes = [cls for cls in classes if len(cls) > 0 and not is_random_string(cls)]
                node.attrib['class'] = ' '.join(new_classes)
                if len(node.attrib['class']) <= 0:
                    del node.attrib['class']
            else:
                # If an attribute is random, remove it
                if len(value) > 0 and is_random_string(value):
                    del node.attrib[attr]
    return node


def remove_random_classes(html_string):
    parser = etree.HTMLParser()
    tree = etree.fromstring(html_string, parser)
    parent = tree
    for child in tree.getchildren():
        for grandchild in child.getchildren():
            parent = process_node_class(grandchild)
    modified_html = etree.tostring(parent, pretty_print=True, method='html').decode('utf-8')
    return modified_html


def process_node(node):
    # Process children first
    for child in node:
        process_node(child)
    # If current node is not the top-level HTML element and it's not 'img', 'i' or 'option'
    # We remove it but keep its text and children
    parent = node.getparent()
    if node.text:
        if parent.text:
            parent.text = '{} {}'.format(parent.text.strip(), node.text.strip())
        else:
            parent.text = node.text.strip()
    if node.tag not in ['img', 'i']:
        # Move its children to its parent
        index = parent.index(node)
        for child in node:
            parent.insert(index, child)
            index += 1
        # Move its tail text to the previous sibling (if any) or to the parent
        if node.tail:
            prev_sibling = node.getprevious()
            if prev_sibling is not None:
                if prev_sibling.tail:
                    prev_sibling.tail = '{} {}'.format(prev_sibling.tail.strip(), node.tail.strip())
                else:
                    prev_sibling.tail = node.tail.strip()
            else:
                if parent.text:
                    parent.text = '{} {}'.format(parent.text.strip(), node.tail.strip())
                else:
                    parent.text = node.tail.strip()
        # Now it's safe to remove the node
        parent.remove(node)
    return parent


def remove_intermediate_elements(html_string):
    parser = etree.HTMLParser()
    tree = etree.fromstring(html_string, parser)
    parent = tree
    for child in tree.getchildren():
        for grandchild in child.getchildren():
            for greatgrandchild in grandchild.getchildren():
                parent = process_node(greatgrandchild)
    modified_html = etree.tostring(parent, pretty_print=True, method='html').decode('utf-8')
    return modified_html


def remove_long_attributes(element, threshold=200):
    parser = etree.HTMLParser()
    tree = etree.fromstring(element, parser)
    for child in tree.getchildren():
        for grandchild in child.getchildren():
            for attrib_name, attrib_value in grandchild.attrib.items():
                # Check the length of the attribute value
                if len(attrib_value) > threshold:
                    del grandchild.attrib[attrib_name]
    modified_html = etree.tostring(grandchild, encoding='unicode')
    return modified_html


def clean_element(element):
    cleaned_html = CLEANER.clean_html(element)
    cleaned_html = remove_intermediate_elements(cleaned_html)
    cleaned_html = remove_long_attributes(cleaned_html)
    cleaned_html = remove_random_classes(cleaned_html)
    return cleaned_html
    # return remove_random_classes(remove_long_attributes(remove_intermediate_elements(CLEANER.clean_html(element))))
