from typing import List, Optional

import numpy as np
import pandas as pd


def process_ocn(filedir: str, num_active_users: Optional[int] = None):
    r""" It processes the 'online social network' dataset. For more info regarding the dataset, please checkout
    https://toreopsahl.com/datasets/#online_social_network, Network1.

    Args:
        filedir (str): Directory to process the text file.
    """

    df = pd.read_csv(filedir, sep=" ", header=None)
    df.columns = ["Time", "Sender", "Receiver", "Completed"]
    
    # Always drop completed column. It  is not needed for our purposes here.
    df.drop(axis=1, labels=["Completed"], inplace=True)
    
    # Generate 'Date' column, which represents the day of interaction, starting from zero (minimum starting date).
    df["Time"] = pd.to_datetime(df["Time"])
    df["Date"] = df["Time"].dt.date
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = (df["Date"] - df["Date"].min()).dt.days
    
    # Drop "Time" as there is no need after building "Date" column.
    df.drop(axis=1, labels=["Time"], inplace=True)

    # Assuming userID starts from zero
    df[["Sender", "Receiver"]] -= 1

    maxID = df[["Sender", "Receiver"]].values.max()
    maxDate = df["Date"].values.max()

    indices = df.to_numpy() 
    if num_active_users:
        num_interactions_per_user = np.zeros(maxID + 1)

        # recID, sendID, _ = np.nonzero(data)
        recID, sendID = indices[:, 0], indices[:, 1]
        np.add.at(num_interactions_per_user, np.concatenate([recID, sendID]), 1)
        activeIDs = num_interactions_per_user.argsort()[-num_active_users:]
        mask = np.logical_or(recID[:, None] == activeIDs[None], sendID[:, None] == activeIDs[None])
        mask = mask.sum(axis=1, dtype=bool)
        indices = indices[mask]

        # Re-index remaining users
        recID, sendID = indices[:, 0], indices[:, 1]
        uID = np.unique(np.concatenate([recID, sendID]))
        uID = np.sort(uID)
        numID = np.arange(uID.size)
        reset_ind = {p: a for p, a in zip(uID, numID)}
        def reindex(x):
            x[0] = reset_ind[x[0]]
            x[1] = reset_ind[x[1]]
            return x
        indices = np.apply_along_axis(reindex, axis=1, arr=indices)

    newMaxID = indices[:, [0, 1]].max()
    data = np.zeros((newMaxID + 1, newMaxID + 1, maxDate + 1))
    data[indices[:,0], indices[:,1], indices[:, 2]] = 1
    data = data[:num_active_users, :num_active_users]
    density = data.sum() / data.size
    print(f"Density of data with shape {data.shape} is {density*100:.3f}%.", flush=True)

    return data

if __name__ == "__main__":
    filedir = "ocnodeslinks.txt"
    filedir = 'http://opsahl.co.uk/tnet/datasets/OCnodeslinks.txt'
    process_ocn(filedir, num_active_users=200)