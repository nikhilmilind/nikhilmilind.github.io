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

    def __init__(self, title, date, url):

        self.title = title
        self.date = date
        self.url = url


def load_pubs(config):

    pubs = list()

    content_dir = config['content_dir']

    for pub_md in os.listdir(f'{content_dir}/publications/'):

        pub_meta, _ = read_markdown(f'{content_dir}/publications/{pub_md}')

        if 'title' not in pub_meta:
            raise ValueError('The title property is required for a page!')
        if 'date' not in pub_meta:
            raise ValueError('The date property is required for a page!')

        pub = pathlib.Path(pub_md).with_suffix('')
        pub_obj = Publication(pub_meta['title'], pub_meta['date'], f'/publications/{pub}.html')
        pubs.append(pub_obj)

    return pubs


def top_pubs(pubs, n=5):

    sorted_pubs = sorted(pubs, key=lambda pub: pub.date, reverse=True)
    return sorted_pubs[:5]


class Post:

    def __init__(self, title, date, url):

        self.title = title
        self.date = date
        self.url = url


def load_posts(config):

    posts = list()

    content_dir = config['content_dir']

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


def top_posts(posts, n=5):

    sorted_posts = sorted(posts, key=lambda post: post.date, reverse=True)
    return sorted_posts[:5]

