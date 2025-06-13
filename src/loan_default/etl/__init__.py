
from pathlib import Path

def get_base_data_dir():
    cur_dir = Path()
    if cur_dir.absolute().stem == "LoanDefault":
        print("in LoanDefault")
        BASE_DATADIR = cur_dir / "data"
    
    elif cur_dir.absolute().stem == "notebooks":
        print("in notebooks")
        BASE_DATADIR = cur_dir.absolute().parent / "data"

    else:
        print(cur_dir.absolute().stem)
        print(f"in {Path().cwd().absolute()}")
        BASE_DATADIR = cur_dir

    return BASE_DATADIR

BASE_DATADIR = get_base_data_dir()