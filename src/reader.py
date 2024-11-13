import pathlib
import re

import yaml
import markdown


def load_config():

    with open('config.yaml', 'r') as config_in:
        config = yaml.safe_load(config_in)
    
    return config


def read_markdown(file_path):

    data = pathlib.Path(file_path).read_text(encoding='utf-8')

    # Parse the YAML frontmatter if it exists
    # Consider the rest to be Markdown
    data_yaml = ''
    data_md = ''
    in_meta_yaml = False
    for line in data.split('\n'):
        if re.match(r'^---$', line):
            in_meta_yaml = not in_meta_yaml
        elif in_meta_yaml:
            data_yaml += line
            data_yaml += '\n'
        else:
            data_md += line
            data_md += '\n'

    md_html = markdown.markdown(data_md)
    md_meta = yaml.safe_load(data_yaml)

    if md_meta is None:
        md_meta = dict()

    return md_meta, md_html
