import numpy as np
import pandas as pd

import zipfile
import os
import sys

def load_data(datasets, seed):
    """
    Takes in dataset list.
    Downloads datasets from UCI repo according
    to specifications in adjoining json.
    Subsamples

    Parameters
    ----------
    datasets : list strings
        List of dataset names, cross-referenced with datasets.json

    Returns
    -------
    Returns a dictionary of datasets.
    {
        'dset': pd.DataFrame,
        'dset': pd.DataFrame
    }
    """
    import requests
    import io
    import json

    datasets_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets.json')
    with open(datasets_json) as j:
        dsets = j.read()
    archive = json.loads(dsets)

    loaded_datasets = {}

    def retrieve_dataset(dataset):
        if dataset['zipped'] == 'f':
            r = requests.post(dataset['url'])
            if r.ok:
                data = r.content.decode('utf8')
                df = pd.read_csv(io.StringIO(data), names=dataset['columns'].split(','), sep=dataset['sep'], index_col=False)
                if dataset['header'] == "t":
                    df = df.iloc[1:]
                return df

            raise "Unable to retrieve dataset: " + dataset
        else:
            r = requests.get(dataset['url'])
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(z.open(dataset['zip_name']), names=dataset['columns'].split(','), sep=dataset['sep'], index_col=False)
            if dataset['header'] == "t":
                df = df.iloc[1:]
            return df

    def select_column(scol):
        # Zero indexed, inclusive
        return scol.split(',')

    def encode_categorical(df, dataset, n_subsample):
        from sklearn.preprocessing import LabelEncoder

        encoders = {}
        if dataset['categorical_columns'] != "":
            for column in select_column(dataset['categorical_columns']):
                encoders[column] = LabelEncoder()
                df[column] = encoders[column].fit_transform(df[column])

        df = df.apply(pd.to_numeric, errors='ignore')
        data_mem = df.memory_usage(index=True).sum()
        df = df.sample(n=n_subsample, random_state=seed)
        print("Memory consumed by " + dataset['name'] + ":" + str(df.memory_usage(index=True).sum()))

        return {"data": df, "target": dataset['target'], "name": dataset['name'], "imbalanced": dataset['imbalanced'], "categorical_columns": dataset['categorical_columns']}

    for d, n_subsample in datasets:
        df = retrieve_dataset(archive[d])
        encoded_df_dict = encode_categorical(df, archive[d], n_subsample) 
        loaded_datasets[d] = encoded_df_dict

    # Return dictionary of pd dataframes
    return loaded_datasets