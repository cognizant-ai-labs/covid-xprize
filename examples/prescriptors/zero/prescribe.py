import argparse



def prescribe(start_date: str,
              end_date: str,
              path_to_hist_file: str,
              output_file_path) -> None:
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-p", "--prior_ips_and_cases",
                        dest="prev_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans and cases")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prev_file, args.output_file)
    print("Done!")
