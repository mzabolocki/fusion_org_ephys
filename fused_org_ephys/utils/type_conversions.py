"""
type_conversions.py

Modules for converting between data types.
"""

###################################################################################################
###################################################################################################


def int_conv(value):
    """
    Check if value can be converted to int.

    Arguments:
    ----------
    value : int, float, str
        Value to be converted to int.

    Returns:
    --------
    value : int
        Value converted to int.
    """

    try:
        value = int(value)
    except (ValueError, TypeError):
        value = None

    return value
