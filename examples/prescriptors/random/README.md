## Example: Random prescriptor

This prescriptor prescribes random valid IPs for each day for each geo.

Example usage:
```
python prescribe.py -s 2020-08-01 -e 2020-08-31 -ip ../../../validation/data/2020-09-30_historical_ip.csv -c ../../../validation/data/uniform_random_costs.csv -o prescriptions/test.csv
```
