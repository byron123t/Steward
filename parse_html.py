import requests
from bs4 import BeautifulSoup


class ParseHtml:
    def __init__(self, html=None, path=None):
        if path is not None:
            with open(path, 'r') as infile:
                self.html = infile.read()
            self.soup = BeautifulSoup(self.html, 'html.parser')            
        elif html is not None:
            self.html = html
            self.soup = BeautifulSoup(self.html, 'html.parser')

    def get_interactables(self, clean=True):
        interactables = []
        interactables.extend(self.soup.select('button'))
        interactables.extend(self.soup.select('a'))
        interactables.extend(self.soup.select('input'))
        interactables.extend(self.soup.select('select'))
        interactables.extend(self.soup.select('textarea'))
        interactables.append(self.soup.select('[role*="{}"]'.format('radio')))
        interactables.append(self.soup.select('[role*="{}"]'.format('option')))
        interactables.append(self.soup.select('[role*="{}"]'.format('checkbox')))
        interactables.append(self.soup.select('[role*="{}"]'.format('button')))
        interactables.append(self.soup.select('[role*="{}"]'.format('tab')))
        interactables.append(self.soup.select('[role*="{}"]'.format('textbox')))
        interactables.append(self.soup.select('[role*="{}"]'.format('link')))
        interactables.append(self.soup.select('[role*="{}"]'.format('menuitem')))
        interactables.append(self.soup.select('[role*="{}"]'.format('tabpanel')))
        interactables.append(self.soup.select('[class*="{}"]'.format('ui-accordion')))
        interactables.append(self.soup.select('[class*="{}"]'.format('ui-draggable-handle')))
        interactables.append(self.soup.select('[class*="{}"]'.format('ui-droppable')))
        interactables.append(self.soup.select('[class*="{}"]'.format('ui-sortable-handle')))
        interactables.append(self.soup.select('[class*="{}"]'.format('ui-accordion-header-collapsed')))
        interactables.append(self.soup.select('[class*="{}"]'.format('ui-datepicker-prev')))
        interactables.append(self.soup.select('[class*="{}"]'.format('ui-datepicker-next')))
        interactables.append(self.soup.select('[class*="{}"]'.format('ui-slider')))
        interactables.extend(self.soup.select('[data-event*=""]'))
        interactables.extend(self.soup.select('[data-action*=""]'))
        interactables.append(self.soup.select('[href]'))
        interactables.append(self.soup.select('[aria-controls]'))
        interactables.append(self.soup.select('[aria-label]'))
        if clean:
            for element in interactables:
                if 'backend_node_id' in element.attrs:
                        del element.attrs['backend_node_id']
        return interactables

    def get_page_text(self):
        return self.soup.get_text(separator=',')
    
    def get_links(self):
        return self.soup.find_all("a")

    def get_tables(self):
        return self.soup.find_all("table")

    def get_buttons(self):
        return self.soup.find_all("button")

    def get_inputs(self):
        return self.soup.find_all("input")
    
    def get_title(self):
        """Scrape page title."""
        title = None
        if self.html.title.string:
            title = self.html.title.string
        elif self.html.find("meta", property="og:title"):
            title = self.html.find("meta", property="og:title").get('content')
        elif self.html.find("meta", property="twitter:title"):
            title = self.html.find("meta", property="twitter:title").get('content')
        elif self.html.find("h1"):
            title = self.html.find("h1").string
        return title

    def get_description(self):
        """Scrape page description."""
        description = None
        if self.html.find("meta", property="description"):
            description = self.html.find("meta", property="description").get('content')
        elif self.html.find("meta", property="og:description"):
            description = self.html.find("meta", property="og:description").get('content')
        elif self.html.find("meta", property="twitter:description"):
            description = self.html.find("meta", property="twitter:description").get('content')
        elif self.html.find("p"):
            description = self.html.find("p").contents
        return description

    def get_site_name(self):
        """Scrape site name."""
        if self.html.find("meta", property="og:site_name"):
            site_name = self.html.find("meta", property="og:site_name").get('content')
        elif self.html.find("meta", property='twitter:title'):
            site_name = self.html.find("meta", property="twitter:title").get('content')
        else:
            site_name = self.url.split('//')[1]
            return site_name.split('/')[0].rsplit('.')[1].capitalize()
        return sitename
