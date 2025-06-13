import chardet
import kaggle as kg
from pathlib import Path
from pandas import DataFrame, read_csv
from typing import Optional
from . import BASE_DATADIR


BASE_DATA_FILENAME = Path("Anonymize_Loan_Default_data.csv")
BASE_FILEPATH = BASE_DATADIR / BASE_DATA_FILENAME

def extract_data_from_kaggle(directory: Optional[Path] = None) -> None:

    """
    Requires API key
    This function uses the Kaggle API to download the dataset onto your device
    If you don't have an API key, you can generate one on your Kaggle profile before running the program.
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

def fix_data_issue(filepath) -> bool:
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        
        if content is not None and content.startswith(','):
            new_content = content[1:]
        if new_content:
            #print("data issue fixed")
            with open(filepath, 'w') as file:
                file.write(new_content)
            return True
        # print("No data issue encountered.")
        return False
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def infer_encoding(filepath):
    with open(filepath, "rb") as file:
        res = chardet.detect(file.read())
    # print(f"encoding inferred: {res["encoding"]}")
    return res

def read_data(filepath: Optional[Path] = None) -> DataFrame:
    #print(filepath)
    if not filepath:
        filepath = BASE_FILEPATH
    if filepath.exists():
        # print(filepath.absolute())
        fix_data_issue(filepath)
        encoding = infer_encoding(filepath)["encoding"]
        return read_csv(filepath, encoding=encoding)
    else:
        # print(filepath.absolute().parent)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        extract_data_from_kaggle(filepath.parent)
        fix_data_issue(filepath)
        encoding = infer_encoding(filepath)["encoding"]
        return read_csv(filepath, delimiter=',', encoding=encoding)

