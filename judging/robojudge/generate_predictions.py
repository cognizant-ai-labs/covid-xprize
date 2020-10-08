"""
Iterates through all sandboxes to generate predictions for each predictor, and uploads the results to S3
"""
import logging.config
import os
import subprocess
from datetime import date

import s3fs

from judging.common.args import parse_args

# Set up logging
from judging.common.constants import Constants

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('robojudge')

# Directory containing links to competitors' sandboxes
SANDBOXES_DIR = 'work'

# Wrapper object for accessing S3
FS = s3fs.S3FileSystem(anon=False)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def generate_predictions(start_date: str, end_date: str, ip_file: str) -> None:
    # enumerate sandboxes
    subdirs = [f.path for f in os.scandir(SANDBOXES_DIR) if f.is_dir()]
    for team in subdirs:
        LOGGER.info(f'Processing: {team}')
        predict_module = f'{team}/predict.py'
        if not os.path.isfile(predict_module):
            LOGGER.warning(f'{predict_module} not found for {team} so cannot process this entry.')
            continue

        # Spawn an external process to run each predictor. In future this may be parallel and even distributed
        subprocess.call(
            [
                'python', predict_module,
                '--start_date', start_date,
                '--end_date', end_date,
                '--interventions_plan', ip_file
            ]
        )


def upload_to_s3(start_date, end_date, prediction_date):
    sub_dirs = [f.path for f in os.scandir(SANDBOXES_DIR) if f.is_dir()]
    prediction_file_name = start_date + "_" + end_date + ".csv"
    for team in sub_dirs:
        predictions_file_path = os.path.join(team, prediction_file_name)
        if not os.path.isfile(predictions_file_path):
            LOGGER.warning(f'Expected predictions for {team} from {start_date} to {end_date} missing: {predictions_file_path}')
            continue

        team = remove_prefix(team, SANDBOXES_DIR + '/')
        s3_destination = f's3://{Constants.S3_BUCKET}/predictions/{prediction_date}/teams/{team}/{prediction_file_name}'

        LOGGER.info(f'Uploading predictions file: {predictions_file_path} to S3: {s3_destination}')
        FS.put(predictions_file_path, s3_destination)


if __name__ == '__main__':
    args = parse_args()
    today_date = date.today().strftime("%Y_%m_%d")
    LOGGER.info(
        f'Generating predictions for {today_date} from start date {args.start_date} to end date {args.end_date}...')
    generate_predictions(args.start_date, args.end_date, args.ip_file)

    upload_to_s3(args.start_date, args.end_date, today_date)
    LOGGER.info(f"Done with predictions for {today_date} from start date {args.start_date} to end date {args.end_date}")
