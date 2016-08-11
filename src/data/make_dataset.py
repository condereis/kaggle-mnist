# -*- coding: utf-8 -*-
import click
import dotenv
import json
import os
import requests
import sys


def kaggle_download(filepath, data_url, kaggle_info):

    # Attempts to download the CSV file. Gets rejected because we are not logged in.
    r = requests.get(data_url)

    # Login to Kaggle and retrieve the data.
    r = requests.post(r.url, data = kaggle_info, stream = True)

    # Writes the data to a local file one chunk at a time.
    with open(filepath, "wb") as f:
        print("Downloading %s" % os.path.basename(filepath))
        total_length = r.headers.get('content-length')
        dl = 0
        for chunk in r.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                if total_length != None: # no content length header
                    dl += len(chunk)
                    total_length = int(total_length)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s] %s%%" % ('=' * done, ' ' * (50-done), 2*done) )    
                    sys.stdout.flush()
        print()


@click.command()
@click.option('--username', prompt='Kaggle Username', type=click.STRING)
@click.option('--password', prompt='Kaggle Password', hide_input=True, type=click.STRING)
def main(username, password):
    kaggle_info = {'UserName': username, 'Password': password}
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    settings = json.loads(open(os.path.join(project_dir, "SETTINGS.json")).read())
    kaggle_download(settings['TRAIN_DATA_PATH'], settings['TRAIN_DATA_URL'], kaggle_info)
    kaggle_download(settings['TEST_DATA_PATH'], settings['TEST_DATA_URL'], kaggle_info)

if __name__ == '__main__':
    main()

