#  Copyright (C) 2023 Battelle Memorial Institute
#  SPDX-License-Identifier: BSD-2-Clause
#  See: https://spdx.org/licenses/

import numpy as np
def sign_extend_24bit(x):
    """
    Sign extends a signed 24-bit numpy array into a signed 32-bit array.
    """
    x = x.astype(np.int32)
    mask = (np.right_shift(x, 23) > 0) * np.array([0xFF000000], dtype=np.int32)
    return np.bitwise_or(x, mask)

def apply_activation(voltage, threshold):
    """Applies a soft-threshold activation"""
    return np.maximum(np.abs(voltage) - threshold, 0) * np.sign(voltage)
