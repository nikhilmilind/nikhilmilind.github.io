import os
import pathlib

from src.reader import read_markdown


class Page:

    def __init__(self, page, name, url):

        self.page = page
        self.name = name
        self.url = url


def load_pages(config):

    pages = list()

    for page in config['pages']:

        content_dir = config['content_dir']
        page_meta, _ = read_markdown(f'{content_dir}/pages/{page}.md')

        if 'name' not in page_meta:
            raise ValueError('The name property is required for a page!')

        page_obj = Page(page, page_meta['name'], f'/{page}.html')
        pages.append(page_obj)

    return pages
        

class Publication:

    def __init__(self, title, preprint, authors, journal, date, doi, url):

        self.title = title
        self.preprint = preprint
        self.authors = authors
        self.journal = journal
        self.date = date
        self.doi = doi
        self.url = url


def load_pubs(config):

    content_dir = config['content_dir']

    if not os.path.isdir(f'{content_dir}/publications/'):
        return list()

    pubs = list()

    for pub_md in os.listdir(f'{content_dir}/publications/'):

        pub_meta, _ = read_markdown(f'{content_dir}/publications/{pub_md}')

        required_properties = ['title', 'preprint', 'authors', 'journal', 'date', 'doi']
        for property in required_properties:
            if property not in pub_meta:
                raise ValueError(f'The {property} property is required for a page!')

        pub_kwargs = {property: pub_meta[property] for property in required_properties}

        pub = pathlib.Path(pub_md).with_suffix('')
        pub_kwargs['url'] = f'/publications/{pub}.html'

        pub_obj = Publication(**pub_kwargs)
        pubs.append(pub_obj)

    return pubs


class Post:

    def __init__(self, title, date, url):

        self.title = title
        self.date = date
        self.url = url


def load_posts(config):

    content_dir = config['content_dir']

    if not os.path.isdir(f'{content_dir}/posts/'):
        return list()

    posts = list()

    for post_md in os.listdir(f'{content_dir}/posts/'):

        post_meta, _ = read_markdown(f'{content_dir}/posts/{post_md}')

        if 'title' not in post_meta:
            raise ValueError('The title property is required for a post!')
        if 'date' not in post_meta:
            raise ValueError('The date property is required for a post!')

        post = pathlib.Path(post_md).with_suffix('')
        post_obj = Post(post_meta['title'], post_meta['date'], f'/posts/{post}.html')
        posts.append(post_obj)

    return posts

