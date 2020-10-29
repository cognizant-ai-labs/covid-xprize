## Example: Zero prescriptor

This prescriptor simply prescribes 0 IPs for each day for each geo. I.e., it says "Do nothing."

Example usage:
```
python prescribe.py -s 2020-08-01 -e 2020-08-31 -ip ../../../validation/data/2020-09-30_historical_ip.csv -c ../../../validation/data/uniform_random_costs.csv -o prescriptions/test.csv
```
