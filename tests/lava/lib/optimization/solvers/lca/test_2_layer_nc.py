# Notice: This computer software was prepared by Battelle Memorial Institute,
# hereinafter the Contractor, under Contract No. DE-AC05-76RL01830 with the 
# Department of Energy (DOE).  All rights in the computer software are reserved
# by DOE on behalf of the United States Government and the Contractor as 
# provided in the Contract.  You are authorized to use this computer software
# for Governmental purposes but it is not to be released or distributed to the
# public.  NEITHER THE GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY,
# EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
# This notice including this sentence must appear on any copies of this 
# computer software.

import unittest

import numpy as np
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io import sink
from lava.proc.embedded_io.spike import NxToPyAdapter
from lava.proc.lif.process import TernaryLIF, LIF

from lava.lib.optimization.solvers.lca.process import LCA2Layer

from lava.lib.optimization.solvers.lca.util import apply_activation

class TestLCALoihi2(unittest.TestCase):
    def test_identity_matrix(self):
        weights = np.eye(5, dtype=np.int8) * 2**8
        weights_exp = -8
        input_val = np.array([6502, 29847, 14746, 8168, 12989])
        threshold = 1
        lca = LCA2Layer(weights=weights, weights_exp=weights_exp,
                        input_vec=input_val, threshold=threshold)
        
        run_config = Loihi2HwCfg(select_sub_proc_model=True)
    
        lca.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        actual = apply_activation(lca.voltage.get(), threshold)
        lca.stop()

        self.assertTrue(np.allclose(input_val, actual, atol=0, rtol=5e-3), 
            f"Expected: {input_val} Actual: {actual}")
        

    def test_negative_residual(self):
        weights = np.eye(5, dtype=np.int8) * 2**8
        weights_exp = -8
        input_val = np.array([26366, -18082, 5808, -10212, -25449])
        threshold = 1
        lca = LCA2Layer(weights=weights, weights_exp=weights_exp,
                        input_vec=input_val, threshold=threshold)

        run_config = Loihi2HwCfg(select_sub_proc_model=True)
    
        lca.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        actual = apply_activation(lca.voltage.get(), threshold)

        lca.stop()

        self.assertTrue(np.allclose(input_val, actual, atol=0, rtol=5e-3), 
            f"Expected: {input_val} Actual: {actual}")
    
    def test_competition(self):
        weights = np.array([[0, np.sqrt(1/2), np.sqrt(1/2)], 
                            [np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)]]) * 2**8
        weights_exp = -8
        input_val = np.array([0, 2**22, 2**22])
        expected = np.array([2**22 / np.sqrt(1/2), 0])

        threshold = 1
        lca = LCA2Layer(weights=weights, weights_exp=weights_exp,
                        input_vec=input_val, threshold=threshold)

        run_config = Loihi2HwCfg(select_sub_proc_model=True)
    
        lca.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        actual = apply_activation(lca.voltage.get(), threshold)

        lca.stop()

        self.assertTrue(np.allclose(expected, actual, atol=15, rtol=1e-3), 
            f"Expected: {expected} Actual: {actual}")
        

    def test_excitation(self):
        weights = np.array([[-np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)],
                            [1, 0, 0]]) * 2**8
        weights_exp = -8
        input_val = np.array([0, 2**22, 2**22])
        expected = np.array([2**22 / np.sqrt(1/3), 2**22])

        threshold = 1
        lca = LCA2Layer(weights=weights, weights_exp=weights_exp,
                        input_vec=input_val, threshold=threshold)

        run_config = Loihi2HwCfg(select_sub_proc_model=True)
        
        lca.run(condition=RunSteps(num_steps=1000), run_cfg=run_config)

        actual = apply_activation(lca.voltage.get(), threshold)

        lca.stop()

        # TODO: see if we can get a closer solution. 
        self.assertTrue(np.allclose(expected, actual, atol=0, rtol=2e-1), 
            f"Expected: {expected} Actual: {actual}")
        

if __name__ == "__main__":
    unittest.main()