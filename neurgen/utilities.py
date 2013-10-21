""" some basic utilities for use with Grammatical Evolution"""

import random

LARGEVALUE_LIMIT = 1e+10
NEGVALUE_LIMIT = -1e+10


def rand_int(start_pos, end_pos):
    """
    This method returns a random number between startposition and length.

    """

    if isinstance(start_pos, int) and isinstance(end_pos, int):
        return random.randint(start_pos, end_pos)
    else:
        raise ValueError("Must be int")


def rand_weight(constraint=1.0):
    """
    Returns a random weight centered around 0.  The constrain limits
    the value to the maximum that it would return.  For example, if
    .5 is the constraint, then the returned weight would between -.5 and +.5.
    """
    return random.random() * 2.0 * constraint - constraint


def rand_value():
    """
    This function returns a random value between 0 and 1.

    """

    return random.random()


def base10tobase2(value, zfill=0):
    """
    This function converts from base 10 to base 2 in string format.  In
    addition, it takes a parameter for zero filling to pad out to a specific
    length.

    Note that the incoming number is converted to an int, and that if it is
    a negative number that the negative sign is added on to the total length,
    resulting in a string 1 char longer than the zfill specified.

    """
    new_value = []
    val = int(value)
    if val < 0:
        neg = True
        val *= -1
    else:
        neg = False

    if val == 0:
        new_value = ['0']

    while val > 0:
        new_value.append(str(val % 2))
        val = val / 2

    new_value.reverse()
    new_value_str = ''.join(new_value)
    if zfill:
        if len(new_value_str) > zfill:
            raise ValueError("""
            Base 2 version of %s is longer, %s, than the zfill limit, %s
            """ % (value, new_value_str, zfill))
        else:
            new_value_str = new_value_str.zfill(zfill)

    if neg:
        new_value_str = "-" + new_value_str

    return new_value_str


def base2tobase10(value):
    """
    This function converts from base 2 to base 10.  Unlike base10tobase2,
    there is no zfill option, and the result is output as an int.

    """

    new_value = 0
    val = str(value)
    if val < 0:
        neg = True
        val *= -1
    else:
        neg = False

    val = str(value)

    factor = 0
    for i in range(len(val) - 1, -1, -1):
        if not val[i] == '-':
            new_value += int(val[i]) * pow(2, factor)
        else:
            neg = True
        factor += 1

    if neg:
        new_value *= -1

    return new_value


def largevalue_limit(value, halt_on_extremes):
    """
    This function checks for out of bounds errors on the large side.

    """

    if value > LARGEVALUE_LIMIT:
        value = LARGEVALUE_LIMIT
        if halt_on_extremes:
            raise ValueError("Halted on extremely large value %s" % value)
    return value


def negvalue_limit(value, halt_on_extremes):
    """
    This function checks for out of bounds errors on the negative side.

    """

    if value < NEGVALUE_LIMIT:
        value = NEGVALUE_LIMIT
        if halt_on_extremes:
            raise ValueError("Halted on extremely negative value %s" % value)
    return value


def check_bounds_limits(value, halt_on_extremes):
    """
    This function checks for out of bounds errors.

    """

    return largevalue_limit(negvalue_limit(value, halt_on_extremes),
            halt_on_extremes)


def string_to_valuelist(value):
    """
    This function converts a list of string based values to a list of floats.

    """

    return [float(i) for i in value]


def zero_or_one(value, cutoff=.50):
    """
    This function accepts a value for use in rounding to 0 or 1.  The cutoff
    value is used as the basis for making the choice.  The value can be either
    a list or a single value.

    """

    if isinstance(value, list):

        newactual = []
        for item in value:
            if item > cutoff:
                newactual.append(1)
            else:
                newactual.append(0)

    elif isinstance(value, float):
        if value > cutoff:
            newactual = 1
        else:
            newactual = 0

    else:
        raise ValueError("Value must be either a list or a float")

    return newactual
