#From https://www.kaggle.com/chenjx1005/two-sigma-financial-modeling/physical-meanings-of-technical-20-30


# Here's an example of loading the CSV using Pandas's built-in HDF5 support:
import pandas as pd

with pd.HDFStore("C:\\Users\\Josh\\Documents\\train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")

    print len(df) # 1,710,756
    print df.head()

    print(len(df["timestamp"].unique())) # 1,813 distinct timestamps
