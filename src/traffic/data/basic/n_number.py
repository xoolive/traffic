# The original implementation comes from the following repository
# https://github.com/guillaumemichel/icao-nnumber-converter

# Amended version includes fixes, optimisations, code simplifications, tests,
# types annotations, mypy fixes and formatting.
# In the end, it's nearly a full rewrite.

from __future__ import annotations

import re
import string
from itertools import count

ICAO_SIZE = 6  # size of an icao address
NNUMBER_MAX_SIZE = 6  # max size of a N-Number

charset = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # alphabet without I and O

# First digit, second digit, third digit, fourth digit
factors = [101711, 10111, 951, 35]
# 35 = 1 + len(charset) + 10
# 601 = all possible suffix offsets
# 951 = 35 * 10 + 601
# 10111 = 951 * 10 + 601
# 101711 = 10111 * 10 + 601

n_pattern = re.compile(r"^N[1-9]\d{0,3}[0-9A-HJ-NP-Z]{0,2}$")
a_pattern = re.compile(r"^a[0-9a-f]{5}$")
split_pattern = re.compile(r"^N(\d{1,4})([A-HJ-NP-Z]{0,2})(\d?)$")


def create_icao(number: int, prefix: str = "a") -> str:
    """
    Creates an American ICAO number composed from the prefix ('a' for USA)
    and from the given number.

    The output is an hexadecimal of length 6 starting with the suffix

    >>> create_icao(11)
    'a0000b'
    >>> create_icao(915399)
    'adf7c7'
    >>> create_icao(999999)
    Traceback (most recent call last):
        ...
    RuntimeError: Invalid value, must be below 0xdf7c7 for N99999

    """
    if number > 0xDF7C7:
        raise RuntimeError("Invalid value, must be below 0xdf7c7 for N99999")

    return prefix + hex(number)[2:].rjust(5, "0")


def get_suffix(offset: int) -> str:
    """Compute the suffix for the tail number given an offset

    An offset of 0 returns in a valid empty suffix
    A non-zero offset return a string containing one or two characters

    >>> get_suffix(0)
    ''
    >>> get_suffix(1)
    'A'
    >>> get_suffix(2)
    'AA'
    >>> get_suffix(3)
    'AB'
    >>> get_suffix(24)
    'AY'
    >>> get_suffix(26)
    'B'
    >>> get_suffix(600)
    'ZZ'

    """
    if offset == 0:
        return ""
    assert offset <= 600, "offset value must be below 600 for ZZ"
    char0 = charset[int((offset - 1) / 25)]
    rem = (offset - 1) % 25
    return char0 + (charset[rem - 1] if rem > 0 else "")


def suffix_offset(suffix: str) -> int:
    """Compute the offset corresponding to the given alphabetical suffix.

    >>> suffix_offset("")
    0
    >>> suffix_offset("A")
    1
    >>> suffix_offset("AA")
    2
    >>> suffix_offset("AZ")
    25
    >>> suffix_offset("BA")
    27
    >>> suffix_offset("ZZ")
    600
    """
    length = len(suffix)

    if length == 0:
        return 0

    if length > 2:
        msg = (
            "Invalid input value:"
            " the suffix must be comprised of at most 2 letters"
        )
        raise RuntimeError(msg)

    if any(c not in charset for c in suffix):
        msg = f"Invalid input value: each letter must be in {charset}"
        raise RuntimeError(msg)

    count = 25 * charset.index(suffix[0]) + 1

    if length == 2:
        count += charset.index(suffix[1]) + 1

    return count


def n_number_to_icao(n_number: str) -> str:
    """Convert a N-number to corresponding a- icao address.

    >>> n_number_to_icao("N1")
    'a00001'
    >>> n_number_to_icao("N1AZ")
    'a0001a'
    >>> n_number_to_icao("N1B")
    'a0001b'
    >>> n_number_to_icao("N1000Z")
    'a00724'
    >>> n_number_to_icao("N10000")
    'a00725'
    >>> n_number_to_icao("N1002")
    'a00752'
    >>> n_number_to_icao("N102A")
    'a00c22'
    >>> n_number_to_icao("N9A")
    'ac6a7a'
    >>> n_number_to_icao("N99999")
    'adf7c7'

    """

    # avoid unnecessary issues
    n_number = n_number.upper()

    if not n_pattern.match(n_number):
        raise RuntimeError(f"{n_number} is not a valid N number")

    m = split_pattern.match(n_number)
    assert m
    digits, offset, final = m.group(1), m.group(2), m.group(3)

    integer = 1
    for i, nb, factor in zip(count(), digits, factors):
        if i == 0:
            integer += factor * (int(nb) - 1)
        else:
            integer += factor * int(nb) + 601

    if len(offset) == 1 and i == 3:
        # Corner case when the Nxxx0A (6 chars, finishes with digit-letter)
        integer += 1 + charset.index(offset)
    else:
        integer += suffix_offset(offset)

    if final != "":
        integer += 25 + int(final)

    return create_icao(integer)


def icao_to_n_number(icao: str) -> None | str:
    """Convert an a- ICAO address to a N-number registration

    >>> icao_to_n_number("a00001")
    'N1'
    >>> icao_to_n_number("a0001a")
    'N1AZ'
    >>> icao_to_n_number("a0001b")
    'N1B'
    >>> icao_to_n_number("a00724")
    'N1000Z'
    >>> icao_to_n_number("a00725")
    'N10000'
    >>> icao_to_n_number("a00752")
    'N1002'
    >>> icao_to_n_number("a00c22")
    'N102A'
    >>> icao_to_n_number("ac6a7a")
    'N9A'
    >>> icao_to_n_number("adf7c7")
    'N99999'
    """

    icao = icao.lower()

    if not a_pattern.match(icao):
        raise RuntimeError(f"{icao} is not a valid US register ICAO address")

    output = "N"  # digit 0 = N

    value = int(icao[1:], base=16) - 1  # parse icao to int
    if value < 0:
        return output

    dig1 = int(value / factors[0]) + 1  # digit 1
    rem1 = value % factors[0]
    output += str(dig1)

    if rem1 < 601:
        return output + get_suffix(rem1)

    rem1 -= 601  # shift for digit 2
    dig2 = int(rem1 / factors[1])
    rem2 = rem1 % factors[1]
    output += str(dig2)

    if rem2 < 601:
        return output + get_suffix(rem2)

    rem2 -= 601  # shift for digit 3
    dig3 = int(rem2 / factors[2])
    rem3 = rem2 % factors[2]
    output += str(dig3)

    if rem3 < 601:
        return output + get_suffix(rem3)

    rem3 -= 601  # shift for digit 4
    dig4 = int(rem3 / factors[3])
    rem4 = rem3 % factors[3]
    output += str(dig4)

    if rem4 == 0:
        return output

    # find last character
    return output + (charset + string.digits)[rem4 - 1]
