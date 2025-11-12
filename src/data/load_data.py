import pandas as pd
from pathlib import Path

def load_sms_spam(csv_path: str):
    """Load the sms spam dataset. Expects two columns: label,text or similar.

    Returns: pandas DataFrame with columns ['label','text']
    """
    p = Path(csv_path)
    df = pd.read_csv(p, header=None)
    # common dataset format: first column label, second column text
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ['label', 'text']
    else:
        raise ValueError('Unexpected CSV format: need at least 2 columns')
    return df
