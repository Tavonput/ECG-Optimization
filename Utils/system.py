import os
import logging

LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("SYS__").setLevel(logging.INFO)
log = logging.getLogger("SYS__")


def check_path_for_dir(path: str, create: bool) -> bool:
    """
    Check the directory of a path.

    Parameters
    ----------
    path : str
        The path to check.
    create : bool
        Whether or not to create the directory if it does not exist.

    Returns
    -------
    success : bool
        Whether or not the directory exists.
    """
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        if create is True:
            log.info(f"Creating directory {path_dir}")
            os.makedirs(path_dir)
            return True
        else:
            log.error(f"{path_dir} does not exist")
            return False
     
    return True


def check_path_exists(path: str) -> bool:
    """
    Wrapper around os.path.exists.

    Parameters
    ----------
    path : str
        The path to check.

    Returns
    -------
    success : bool
        Whether or not the path exists.
    """
    if not os.path.exists(path):
        log.error(f"{path} does not exist")
        return False

    return True