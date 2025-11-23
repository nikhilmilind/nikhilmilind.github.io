from src.reader import load_config, read_markdown
from src.models import load_pages, load_pubs, load_posts
from src.filters import pub_author_fmt

import os
import pathlib
import shutil

from jinja2 import Environment, FileSystemLoader


def render_post(config, base_context, env, post_md):

    post_template = config['post_template']
    template = env.get_template(post_template)
    
    content_dir = config['content_dir']
    post_meta, post_content = read_markdown(f'{content_dir}/posts/{post_md}')

    context = dict()
    context['config'] = config
    context['meta'] = post_meta
    context = context | base_context
    context['content_post'] = post_content

    rendered_html = template.render(context)

    output_dir = config['output_dir']
    post = pathlib.Path(post_md).with_suffix('')
    with open(f'{output_dir}/posts/{post}.html', 'w') as f_out:
        f_out.write(rendered_html)


def render_pub(config, base_context, env, pub_md):

    pub_template = config['pub_template']
    template = env.get_template(pub_template)
    
    content_dir = config['content_dir']
    pub_meta, pub_content = read_markdown(f'{content_dir}/publications/{pub_md}')

    context = dict()
    context['config'] = config
    context['meta'] = pub_meta
    context = context | base_context
    context['content_pub'] = pub_content

    rendered_html = template.render(context)

    output_dir = config['output_dir']
    pub = pathlib.Path(pub_md).with_suffix('')
    with open(f'{output_dir}/publications/{pub}.html', 'w') as f_out:
        f_out.write(rendered_html)


def render_publications(config, base_context, env):

    publications_template = config['publications_template']
    template = env.get_template(publications_template)

    context = dict()
    context['config'] = config
    context = context | base_context
    context['page'] = 'publications'

    rendered_html = template.render(context)

    output_dir = config['output_dir']
    with open(f'{output_dir}/{publications_template}', 'w') as f_out:
        f_out.write(rendered_html)


def render_page(config, base_context, env, page):

    page_template = config['page_template']
    template = env.get_template(page_template)
    
    content_dir = config['content_dir']
    page_meta, page_content = read_markdown(f'{content_dir}/pages/{page}.md')

    context = dict()
    context['config'] = config
    context['meta'] = page_meta
    context = context | base_context
    context['page'] = page
    context['content_page'] = page_content

    rendered_html = template.render(context)

    output_dir = config['output_dir']
    with open(f'{output_dir}/{page}.html', 'w') as f_out:
        f_out.write(rendered_html)

def render_index(config, base_context, env):

    index_template = config['index_template']
    template = env.get_template(index_template)

    content_dir = config['content_dir']
    index_meta, index_content = read_markdown(f'{content_dir}/profile/index.md')

    context = dict()
    context['config'] = config
    context['meta'] = index_meta
    context = context | base_context
    context['content_index'] = index_content

    rendered_html = template.render(context)

    output_dir = config['output_dir']
    with open(f'{output_dir}/{index_template}', 'w') as f_out:
        f_out.write(rendered_html)


def render_output():

    # Load configuration
    config = load_config()

    # Recreate the output directory
    output_dir = config['output_dir']
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    os.mkdir(f'{output_dir}/publications/')
    os.mkdir(f'{output_dir}/posts/')

    # Copy static files from templates and content
    templates_dir = config['templates_dir']
    shutil.copytree(f'{templates_dir}/static/', f'{output_dir}/static/')

    content_dir = config['content_dir']
    shutil.copytree(f'{content_dir}/static/', f'{output_dir}/static/', dirs_exist_ok=True)

    # Create a new Jinja environment for templating
    env = Environment(loader=FileSystemLoader(templates_dir))
    env.filters['pub_author_fmt'] = pub_author_fmt

    # Create base context
    base_context = dict()
    base_context['pages'] = load_pages(config)
    base_context['pubs'] = load_pubs(config)
    base_context['posts'] = load_posts(config)

    # Render the index page
    render_index(config, base_context, env)

    # Render the publications list page
    render_publications(config, base_context, env)

    # Render all pages
    for page in config['pages']:
        render_page(config, base_context, env, page)

    # Render all publications
    if os.path.isdir(f'{content_dir}/publications/'):
        for pub_md in os.listdir(f'{content_dir}/publications/'):
            render_pub(config, base_context, env, pub_md)

    # Render all posts
    if os.path.isdir(f'{content_dir}/posts/'):
        for post_md in os.listdir(f'{content_dir}/posts/'):
            render_post(config, base_context, env, post_md)
