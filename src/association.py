import math
from functools import reduce
from itertools import product
from typing import List

import pandas as pd


def mutual_information(table, r_margs, k_margs, r, k, n):
    """
    The `mutual information <https://en.wikipedia.org/wiki/Mutual_information>`_ between
    two variables :math:`X` and :math:`Y` is denoted as :math:`I(X;Y)`.  :math:`I(X;Y)` is
    unbounded and in the range :math:`[0, \\infty]`. A higher mutual information
    value implies strong association. The formula for :math:`I(X;Y)` is defined as follows.

    :math:`I(X;Y) = \\sum_y \\sum_x P(x, y) \\log \\frac{P(x, y)}{P(x) P(y)}`

    :return: Mutual information.
    """
    get_p_a = lambda i: r_margs[i] / n
    get_p_b = lambda j: k_margs[j] / n
    get_p_ab = lambda i, j: table[i][j] / n
    get_mi = lambda i, j: get_p_ab(i, j) * math.log(get_p_ab(i, j) / get_p_a(i) / get_p_b(j))

    mi = sum((get_mi(i, j) for i, j in product(*[range(r), range(k)])))

    return mi


def uncertainty_coefficient(contingency_table: List[List]):
    """
    The `uncertainty coefficient <https://en.wikipedia.org/wiki/Uncertainty_coefficient>`_ :math:`U(X|Y)`
    for two variables :math:`X` and :math:`Y` is defined as follows.

    :math:`U(X|Y) = \\frac{I(X;Y)}{H(X)}`

    Where,

    - :math:`H(X) = -\\sum_x P(x) \\log P(x)`
    - :math:`I(X;Y) = \\sum_y \\sum_x P(x, y) \\log \\frac{P(x, y)}{P(x) P(y)}`

    :math:`H(X)` is called the entropy of :math:`X` and :math:`I(X;Y)` is the mutual information
    between :math:`X` and :math:`Y`. Note that :math:`I(X;Y) < H(X)` and both values are positive.
    As such, the uncertainty coefficient may be viewed as the normalized mutual information
    between :math:`X` and :math:`Y` and in the range :math:`[0, 1]`.

    :return: Uncertainty coefficient.
    """
    n_rows = len(contingency_table)
    n_cols = len(contingency_table[0])

    r_margs = [sum(contingency_table[r]) for r in range(n_rows)]
    k_margs = [sum([contingency_table[r][c] for r in range(len(contingency_table))]) for c in range(n_cols)]
    n = sum(r_margs)
    r = len(r_margs)
    k = len(k_margs)

    h_b = map(lambda j: k_margs[j] / n, range(k))
    h_b = map(lambda p: p * math.log(p), h_b)
    h_b = -reduce(lambda x, y: x + y, h_b)

    i_ab = mutual_information(contingency_table, r_margs, k_margs, r, k, n)

    e = i_ab / h_b

    return e


def _get_contingency_table(categorical_column, target_column):
    df = pd.DataFrame({'a': categorical_column, 'b': target_column})
    a_values = sorted([v for v in df.a.unique() if pd.notna(v)])
    b_values = sorted([v for v in df.b.unique() if pd.notna(v)])

    table = [[df.query(f'a=={x} and b=={y}').shape[0] + 1 for y in b_values] for x in a_values]
    return pd.DataFrame(table)
