


# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import kaggle as kg
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "./data/loan-data.csv"

def extract_data():

    # Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "joebeachcapital/loan-default",
    file_path
    # Provide any additional arguments like 
    # sql_query or pandas_kwargs. See the 
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    print("First 5 records:", df.head())
    return df

#kg.api.dataset_download_files(dataset = "joebeachcapital/loan-default", path=file_path, unzip=True)