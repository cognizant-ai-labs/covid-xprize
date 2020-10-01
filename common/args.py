"""
Common code for parsing commonly occurring set of args
"""
import argparse
from argparse import Namespace


def parse_args(with_ip: bool=True) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")

    if with_ip:
        parser.add_argument("-ip", "--interventions_plan",
                            dest="ip_file",
                            type=str,
                            required=True,
                            help="The path to an intervention plan .csv file")
    return parser.parse_args()
