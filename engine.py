from src.reader import load_config, read_markdown
from src.renderer import render_output
from src.server import http_serve

import click


@click.group()
def cli():

    read_markdown('content/profile/index.md')


@cli.command()
def render():

    render_output()


@cli.command()
def serve():

    http_serve()


if __name__ == '__main__':
    
    cli()
