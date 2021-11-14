# -*- coding: utf-8 -*-
import click
import logging
import urllib
from pathlib import Path
import tensorflow_datasets as tfds
from dotenv import find_dotenv, load_dotenv


def store_image(url, local_file_name): 
    with urllib.request.urlopen(url) as resource: 
        with open(local_file_name, 'wb') as f: 
            f.write(resource.read())

def get_wider_face_dataset(info=False): 
    # returns a tf dataset object
    if info == True: 
        ds, info = tfds.load('wider_faces', split=['train', 'test'], shuffle_files=True, with_info=True)
        return ds, info
    ds = tfds.load('wider_faces', split=['train', 'test'], shuffle_files=True)
    return ds
    



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
