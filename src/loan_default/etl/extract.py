
import kaggle as kg
from pathlib import Path
from pandas import DataFrame, read_csv
from typing import Optional


BASE_DATADIR = Path("./data/")
BASE_DATA_FILENAME = Path("Anonymize_Loan_Default_data.csv")
BASE_FILEPATH = BASE_DATADIR / BASE_DATA_FILENAME

def extract_data_from_kaggle(directory: Optional[Path] = None) -> None:

    """
    Requires API key
    (talk about Kaggle, instructions, etc.)

    """

    if not directory:
        directory = BASE_DATADIR

    try:
        kg.api.dataset_download_files(dataset="joebeachcapital/loan-default",
                                           path=directory, 
                                           unzip=True)
    except Exception as e:
        print(f"Make sure you have the API key. Error: {e}")
        raise e


def read_data(filepath: Optional[Path] = None) -> DataFrame:
    if not filepath:
        filepath = BASE_FILEPATH
    if filepath.exists():
        return read_csv(filepath)
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        extract_data_from_kaggle(filepath.parent)
        return read_csv(filepath, delimiter=',')

