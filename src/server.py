from src.reader import load_config
from src.renderer import render_output

import http.server
import socketserver
import threading
import webbrowser

import requests

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class HttpRequestHandler(http.server.SimpleHTTPRequestHandler):

    '''
    Handle HTTP requests from the local server. Redirects
    requests to files in the output directory. The root 
    URL request is sent to index.html.
    '''

    def do_GET(self):

        config = load_config()
        if self.path == '/':
            self.path = config['index_template']
        output_dir = config['output_dir']
        self.path = f'{output_dir}/{self.path}'

        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    

class FileChangeHandler(FileSystemEventHandler):

    '''
    Handle changes to files in the project while the server
    is running to provide dynamic reloading.
    '''

    def on_any_event(self, event):
        print(f'Modified {event.src_path}')
        try:
            render_output()
        except:
            print('Rendering failed')


def run_server(port):

    '''
    Run the HTTP server using the specified port. This is a
    blocking call.
    '''

    with socketserver.TCPServer(('', port), HttpRequestHandler) as http_server:

        http_server.serve_forever()


def http_serve():

    '''
    Run the HTTP server with autoreload. A new thread is
    created to run the server, while autoreload is run on
    the main thread.
    '''

    # Load configuration
    config = load_config()
    port = config['server_port']

    # Start by rendering the output
    render_output()

    # Run the server on a new thread
    server_thread = threading.Thread(target=run_server, kwargs={'port': port})
    server_thread.daemon = True
    server_thread.start()

    # Handle file changes to autoreload the project
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, 'content/', recursive=True)
    observer.schedule(event_handler, 'templates/', recursive=True)
    observer.start()

    # Open the browser at the index once the server starts
    server_url = f'http://localhost:{port}'
    server_started = False
    while not server_started:
        try:
            response = requests.get(server_url, timeout=5)
            server_started = response.status_code == 200
        except requests.exceptions.RequestException:
            continue

    webbrowser.open_new_tab(server_url)

    # Block the main thread
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print()
        print('Server stopped by user')
        observer.stop()

    observer.join()