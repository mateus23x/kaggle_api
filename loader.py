
"""
    This module provides an OOP abstraction to make loading data easier and focusing on what matters.
"""

# built-in
import os
import pathlib
from urllib.parse import urlparse, quote
import zipfile

# third-party
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


class Data:


    supported_formats = [".txt", ".csv"]

    __slots__ = ("path", "data", "kaggle_api", "download_path")


    def __init__(self, path=None, kaggle_url=None, kaggle_file=None, download_path=None) -> None:

        """
            Class constructor method for 'loader.Data' class.
            This method validate identify whether is a local (file) or a \
            remote (kaggle datasets) loading, validate and prepare for loading. \

            Regarding the kaggle API, please watch this short video to setup your \
            own environment credentials: 'https://youtu.be/DgGFhQmfxHo'.

            Parameters:

                path: path to the file you want to load
                kaggle_url: url that holds the file to load
                kaggle_file: file you want to load from kaggle
                download_path: path to the kaggle API download folder

            Raises:

                ValueError: if the combination of the parameters is valid
            
            Returns: None
        """

        if path is not None and not all([kaggle_url, kaggle_file, download_path]):
            
            # casting path parameter and validating it
            path = pathlib.Path(path)
            self.validate_local_path(path)
            self.path = path

            self.data = self.load(path=path)

        elif all([kaggle_url, kaggle_file, download_path]) and path is None:
            
            # assigning given parameter
            self.download_path = pathlib.Path(download_path).resolve()

            # initializing Kaggle API
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
            
            self.data = self.kaggle_load(url=kaggle_url, file=kaggle_file)
        
        else:
            raise ValueError("Invalid combination of given parameters")


    def validate_local_path(self, path) -> None:
        """
            This function checks if the file is a file and if the file extension is supported.

            Parameters:

                path: The path to the file you want to validate

            Raises:
                
                ValueError: if the given path is invalid
            
            Returns: None
        """

        if not path.is_file():
            raise ValueError("'path' is not a file")

        if path.suffix not in self.supported_formats:
            print("Supported formats", self.supported_formats)
            raise ValueError("Invalid 'path' extension")

        return None


    def load(self, path) -> pd.DataFrame:
        """
            Method that allows to load the data from local files.

            Parameters:

                path: The path to the file to be loaded
            
            Raises:

                NotImplementedError: if the given file format is not supported
            
            Returns: pandas.DataFrame object
        """

        if path.suffix in [".csv", ".txt"]:
            return pd.read_csv(str(path), sep=",")

        elif path.suffix == ".tsv":
            return pd.read_csv(str(path), sep="\t")

        elif path.suffix == ".zip": # kaggle API download file format

            extract_folder = str(path.parent)
            extracted_file_path = pathlib.Path(str(path)[:-len(path.suffix)])

            with zipfile.ZipFile(str(path), "r") as zipref:
                zipref.extractall(extract_folder)

            # recursively import the extracted file
            return self.load(extracted_file_path)

        else:
            raise NotImplementedError(f"load {path.suffix}")


    def kaggle_load(self, url, file) -> pd.DataFrame:
        """
            Method to download a file from Kaggle and returns the data as a pandas.DataFrame.

            Parameters:

                url: The URL of the dataset
                file: The name of the file to download
            
            Raises:

                ValueError: if the given URL doesn't follow the expected pattern.
                RuntimeError: if any error occurs while downloading from kaggle API.

            Returns: pandas.DataFrame object
        """
        
        url_components = urlparse(url)

        if url_components.netloc != "www.kaggle.com":
            raise ValueError(f"Invalid URL netloc. Expected: www.kaggle.com. Given: {url_components.netloc}")

        if url_components.path.startswith("/competitions/"):

            competition_name = url_components.path[14:]
            if competition_name.endswith("/data"):
                competition_name = competition_name[:-5]

            download_path = self.download_path.joinpath(quote(competition_name))
            download_path.mkdir(parents=True, exist_ok=True)

            # download target dataset from competition datasets
            self.kaggle_api.competition_download_file(
                competition=competition_name,
                file_name=file,
                path=str(download_path),
                force=True # overwrite if file already exists
            )

        elif url_components.path.startswith("/datasets/"):
        
            dataset_id = url_components.path[10:]

            dataset_name = dataset_id.split("/")[-1]
            download_path = self.download_path.joinpath(quote(dataset_name))
            download_path.mkdir(parents=True, exist_ok=True)

            # download target dataset from standalone datasets
            if not self.kaggle_api.dataset_download_file(
                dataset=dataset_id,
                file_name=file,
                path=str(download_path),
                force=True # overwrite if file already exists
            ):
                raise RuntimeError("Unable to download standalone dataset")
        
        else:
            raise ValueError(f"Invalid URL path: '{url_components.path}'. The expected value starts with: '/competitions/' or '/datasets/'")

        # the downloaded file comes with an url escaped name
        downloaded_file_path = download_path.joinpath(quote(file))

        # loading the downloaded file
        dataset = self.load(downloaded_file_path)

        return dataset


    def head(self) -> pd.DataFrame:
        """
            Wrapper to the pandas.DataFrame.head method.
            
            Returns: pandas.DataFrame object
        """
        return self.data.head()


def usage():
    """
        This function shows usage examples of the 'loader.Data' class.
    """

    # artificially creating a local file
    path = pathlib.Path(os.path.dirname(__file__))
    path = path.joinpath("data")
    path.mkdir(parents=True, exist_ok=True)
    path = str(path.joinpath("data.csv").resolve())
    df = pd.DataFrame([["1", "2"]], columns=["A", "B"])
    df.to_csv(path, index=False)

    # local file loading example
    path = pathlib.Path(os.path.dirname(__file__))
    path = path.joinpath("data", "data.csv")
    path = str(path.resolve())
    print(f"\nLoading data from '{path}'")
    data = Data(path=path)
    print(data.head())

    download_path = pathlib.Path(os.path.dirname(__file__))
    download_path = download_path.joinpath("..", "temp")
    download_path = str(download_path)

    # kaggle standalone dataset loading example
    url = "https://www.kaggle.com/datasets/gpiosenka/tree-nuts-image-classification"
    file = "tree nuts.csv"
    print(f"\nLoading kaggle dataset:\nURL: '{url}'\nFile: '{file}'")
    data = Data(
        kaggle_url=url,
        kaggle_file=file,
        download_path=download_path
    )
    print(data.head())

    # kaggle competition dataset loading example
    url = "https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data"
    file = "test.tsv.zip"
    print(f"\nLoading kaggle dataset:\nURL: '{url}'\nFile: '{file}'")
    data = Data(kaggle_url=url, kaggle_file=file, download_path=download_path)
    print(data.head())

    return None
