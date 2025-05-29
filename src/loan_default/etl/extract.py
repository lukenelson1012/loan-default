
import kaggle as kg


file_path = "./data/"
kg.api.dataset_download_files(dataset = "joebeachcapital/loan-default", path=file_path, unzip=True)
