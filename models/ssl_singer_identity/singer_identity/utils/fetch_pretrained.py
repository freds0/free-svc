"""Downloads or otherwise fetches pretrained models
 Almost entirely adapted from https://github.com/speechbrain/speechbrain/blob/f02eafc1e8ac3f094e89a4c31941e44f4dbcab62/speechbrain/pretrained/fetching.py

"""
import urllib.request
import urllib.error
import pathlib
import logging
from enum import Enum
import huggingface_hub
from typing import Union
from collections import namedtuple
from requests.exceptions import HTTPError
import hashlib
import sys
import yaml
import torch


logger = logging.getLogger(__name__)


def _missing_ok_unlink(path):
    # missing_ok=True was added to Path.unlink() in Python 3.8
    # This does the same.
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class FetchFrom(Enum):
    """Designator where to fetch models/audios from.

    Note: HuggingFace repository sources and local folder sources may be confused if their source type is undefined.
    """

    LOCAL = 1
    HUGGING_FACE = 2
    URI = 3


# For easier use
FetchSource = namedtuple("FetchSource", ["FetchFrom", "path"])
FetchSource.__doc__ = """NamedTuple describing a source path and how to fetch it"""
FetchSource.__hash__ = lambda self: hash(self.path)
FetchSource.encode = lambda self, *args, **kwargs: "_".join(
    (str(self.path), str(self.FetchFrom))
).encode(*args, **kwargs)
# FetchSource.__str__ = lambda self: str(self.path)


def fetch(
    filename,
    source,
    savedir="./pretrained_model_checkpoints",
    overwrite=False,
    save_filename=None,
    use_auth_token=False,
    revision=None,
    cache_dir: Union[str, pathlib.Path, None] = None,
    silent_local_fetch: bool = False,
):
    """Ensures you have a local copy of the file, returns its path

    In case the source is an external location, downloads the file.  In case
    the source is already accessible on the filesystem, creates a symlink in
    the savedir. Thus, the side effects of this function always look similar:
    savedir/save_filename can be used to access the file. And save_filename
    defaults to the filename arg.

    Arguments
    ---------
    filename : str
        Name of the file including extensions.
    source : str or FetchSource
        Where to look for the file. This is interpreted in special ways:
        First, if the source begins with "http://" or "https://", it is
        interpreted as a web address and the file is downloaded.
        Second, if the source is a valid directory path, a symlink is
        created to the file.
        Otherwise, the source is interpreted as a Huggingface model hub ID, and
        the file is downloaded from there.
    savedir : str
        Path where to save downloads/symlinks.
    overwrite : bool
        If True, always overwrite existing savedir/filename file and download
        or recreate the link. If False (as by default), if savedir/filename
        exists, assume it is correct and don't download/relink. Note that
        Huggingface local cache is always used - with overwrite=True we just
        relink from the local cache.
    save_filename : str
        The filename to use for saving this file. Defaults to filename if not
        given.
    use_auth_token : bool (default: False)
        If true Hugginface's auth_token will be used to load private models from the HuggingFace Hub,
        default is False because majority of models are public.
    revision : str
        The model revision corresponding to the HuggingFace Hub model revision.
        This is particularly useful if you wish to pin your code to a particular
        version of a model hosted at HuggingFace.
    cache_dir: str or Path (default: None)
        Location of HuggingFace cache for storing pre-trained models, to which symlinks are created.
    silent_local_fetch: bool (default: False)
        Surpress logging messages (quiet mode).

    Returns
    -------
    pathlib.Path
        Path to file on local file system.

    Raises
    ------
    ValueError
        If file is not found
    """
    if save_filename is None:
        save_filename = filename
    savedir = pathlib.Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    fetch_from = None
    if isinstance(source, FetchSource):
        fetch_from, source = source
    sourcefile = f"{source}/{filename}"
    if pathlib.Path(source).is_dir() and fetch_from not in [
        FetchFrom.HUGGING_FACE,
        FetchFrom.URI,
    ]:
        # Interpret source as local directory path & return it as destination
        sourcepath = pathlib.Path(sourcefile).absolute()
        MSG = f"Destination {filename}: local file in {str(sourcepath)}."
        if not silent_local_fetch:
            logger.info(MSG)
        return sourcepath
    destination = savedir / save_filename
    if destination.exists() and not overwrite:
        MSG = f"Fetch {filename}: Using existing file/symlink in {str(destination)}."
        logger.info(MSG)
        return destination
    if (
        str(source).startswith("http:") or str(source).startswith("https:")
    ) or fetch_from is FetchFrom.URI:
        # Interpret source as web address.
        MSG = f"Fetch {filename}: Downloading from normal URL {str(sourcefile)}."
        logger.info(MSG)
        # Download
        try:
            urllib.request.urlretrieve(sourcefile, destination)
        except urllib.error.URLError:
            raise ValueError(
                f"Interpreted {source} as web address, but could not download."
            )
    else:  # FetchFrom.HUGGING_FACE check is spared (no other option right now)
        # Interpret source as huggingface hub ID
        # Use huggingface hub's fancy cached download.
        MSG = f"Fetch {filename}: Delegating to Huggingface hub, source {str(source)}."
        print(MSG)
        logger.info(MSG)
        try:
            fetched_file = huggingface_hub.hf_hub_download(
                repo_id=source,
                filename=filename,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
            )
            logger.info(f"HF fetch: {fetched_file}")
        except HTTPError as e:
            if "404 Client Error" in str(e):
                raise ValueError("File not found on HF hub")
            else:
                raise

        # Huggingface hub downloads to etag filename, symlink to the expected one:
        sourcepath = pathlib.Path(fetched_file).absolute()
        # Create destination directory if it does not exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        _missing_ok_unlink(destination)
        destination.symlink_to(sourcepath)
    return destination


def from_hparams(
    cls,
    source,
    hparams_file="hyperparams.yaml",
    weights_file="model.pt",
    pymodule_file="custom.py",
    overrides={},
    savedir=None,
    use_auth_token=False,
    revision=None,
    download_only=False,
    **kwargs,
):
    """Fetch and load based from outside source based on HyperPyYAML file

    The source can be a location on the filesystem or online/huggingface

    You can use the pymodule_file to include any custom implementations
    that are needed: if that file exists, then its location is added to
    sys.path before Hyperparams YAML is loaded, so it can be referenced
    in the YAML.

    The hyperparams file should contain a "modules" key, which is a
    dictionary of torch modules used for computation.

    The hyperparams file should contain a "pretrainer" key, which is a
    speechbrain.utils.parameter_transfer.Pretrainer

    Adapted from https://github.com/speechbrain/

    """
    if savedir is None:
        clsname = cls.__name__
        savedir = f"./pretrained_models/{clsname}-{hashlib.md5(source.encode('UTF-8', errors='replace')).hexdigest()}"
    hparams_local_path = fetch(
        filename=hparams_file,
        source=source,
        savedir=savedir,
        overwrite=False,
        save_filename=None,
        use_auth_token=use_auth_token,
        revision=revision,
    )
    weights_local_path = fetch(
        filename=weights_file,
        source=source,
        savedir=savedir,
        overwrite=False,
        save_filename=None,
        use_auth_token=use_auth_token,
        revision=revision,
    )

    try:
        pymodule_local_path = fetch(
            filename=pymodule_file,
            source=source,
            savedir=savedir,
            overwrite=False,
            save_filename=None,
            use_auth_token=use_auth_token,
            revision=revision,
        )
        sys.path.append(str(pymodule_local_path.parent))
    except ValueError:
        if pymodule_file == "custom.py":
            # The optional custom Python module file did not exist
            # and had the default name
            pass
        else:
            # Custom Python module file not found, but some other
            # filename than the default was given.
            raise

    # Load the modules:
    # with open(hparams_local_path) as fin:
    #     hparams = load_hyperpyyaml(fin, overrides)

    hparams = yaml.safe_load(open(hparams_local_path, "r"))

    # Load on the CPU. Later the params can be moved elsewhere by specifying
    if not download_only:
        # Now return the system
        model_class = cls(**hparams, **kwargs)
        model_class.load_state_dict(torch.load(weights_local_path, map_location="cpu"))
        print("Model loaded from", weights_local_path)
        return model_class


def from_scripted(filename, source, savedir=None):
    """Load a model from a scripted file"""
    if savedir is None:
        savedir = f"./pretrained_models/{filename}-{hashlib.md5(source.encode('UTF-8', errors='replace')).hexdigest()}"
    filename = filename + ".ts" if not filename.endswith(".ts") else filename

    print(filename, source, savedir)
    model_file = fetch(
        filename=filename,
        source=source,
        savedir=savedir,
    )

    model = torch.jit.load(model_file)
    print("Model loaded from", model_file)
    return model
