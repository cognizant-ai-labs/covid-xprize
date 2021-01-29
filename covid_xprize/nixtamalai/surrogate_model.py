import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = glob(os.path.join(ROOT_DIR, "..", "..", "prescriptions/*.csv"))
FILES.sort()

prescriptions = {os.path.basename(fname).split("-")[0]: pd.read_csv(fname, parse_dates=["Date"], index_col=["Date"])
                 for fname in tqdm(FILES)}

presc_norm = {k: v - prescriptions['332423242324'] for k, v in prescriptions.items()}
presc_norm = {k: v.sum() for k, v in presc_norm.items()}

presc_norm_df = pd.DataFrame(presc_norm).T

presc_norm_df.loc["332423242324"] = 1
presc_norm_df = np.log(presc_norm_df)

presc_norm_df.to_csv("presc-cases.csv")