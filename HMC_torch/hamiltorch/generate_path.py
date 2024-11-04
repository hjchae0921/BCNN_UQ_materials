def generate_path(base_path, iterator=None):
    """
    Generates a new path format by inserting the iterator before the `.pkl`.
    
    Args:
    - base_path (str): Original path of the format `.../params_hmc.pkl`
    - iterator (int, optional): Index to be inserted before `.pkl`. If not provided, returns the base format.
    
    Returns:
    - str: New path format
    """
    
    if iterator is None:
        return base_path

    # Split the path to insert the iterator
    prefix, ext = os.path.splitext(base_path)
    return f"{prefix}_{iterator}{ext}"