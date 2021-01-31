from covid_xprize.nixtamalai import analyze_predictor
from covid_xprize.nixtamalai.analyze_predictor import IP_MAX_VALUES, IP_COLS
import sys
import os
import numpy as np
from collections import OrderedDict


def random_prescriptions():
    prescriptions = OrderedDict()
    prescriptions["0" * len(IP_MAX_VALUES)] = True
    prescriptions["".join(map(str, [IP_MAX_VALUES[k] for k in analyze_predictor.IP_COLS]))] = True
    for j in range(len(IP_MAX_VALUES)):
        pres = [IP_MAX_VALUES[k] for k in analyze_predictor.IP_COLS]
        pres[j] = 0
        prescriptions["".join(map(str, pres))] = True
    while len(prescriptions) < 1024:
        _ = [np.random.randint(IP_MAX_VALUES[k] + 1) for k in analyze_predictor.IP_COLS]
        key = "".join(map(str, _))
        if key not in prescriptions:
            prescriptions[key] = True
    with open("prescriptions.txt", "w") as fpt:
        [print(x, file=fpt) for x in prescriptions]


if __name__ == "__main__":
    with open("prescriptions.txt") as fpt:
        l = [x.strip() for x in fpt.readlines()]

    n = int(sys.argv[1])
    date = "2020-08-01"
    dd = analyze_predictor.cases(date, 89, l[n])

    output = os.path.join("prescriptions", l[n] + "-" + date + ".csv")
    dd.to_csv(output)
