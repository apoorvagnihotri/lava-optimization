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

from lava.proc.io import sink, source
from lava.proc.dense.process import Dense
from lava.proc.embedded_io.spike import NxToPyAdapter, PyToNxAdapter
from lava.magma.core.run_conditions import RunSteps

from lava.lib.optimization.solvers.lca.v1_neuron.process import \
    V1Neuron

from lava.lib.optimization.solvers.lca.util import sign_extend_24bit
from lava.magma.core.run_configs import Loihi2HwCfg

class TestV1Neuron(unittest.TestCase):
    def test_compilation(self):
        input_source = source.RingBuffer(data=np.array([[8, 7, 0, 8, 7]]).T)
        output_sink = sink.RingBuffer(shape=(5,), buffer=1)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        lca_neuron = V1Neuron(shape=(5,), vth=10, tau=0.1, tau_exp=0)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(lca_neuron.a_in)
        lca_neuron.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.compile(run_cfg=run_config)
        
        input_source.stop()

    def test_pos_spikes(self):
        input_data = np.array([[6, 7, 256, 2097152, 1]]).T

        input_source = source.RingBuffer(data=input_data)
        output_sink = sink.RingBuffer(shape=(5,), buffer=5)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        lca_neuron = V1Neuron(shape=(5,), vth=10, tau=1, tau_exp=-3)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(lca_neuron.a_in)
        lca_neuron.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.run(condition=RunSteps(num_steps=5), run_cfg=run_config)

        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 246, 2097142, 0],
                             [1, 3, 501, 4194293, 0],
                             [6, 8, 756, 6291444, 0]
                            ]).T
        
        actual = output_sink.data.get()

        input_source.stop()

        self.assertTrue(np.all(expected == actual), 
            f"Expected: {expected} Actual: {actual}")

    def test_neg_spikes(self):
        input_data = np.array([[-1, 7, 256, -14, -1024]]).T

        input_source = source.RingBuffer(data=input_data)
        output_sink = sink.RingBuffer(shape=(5,), buffer=5)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        lca_neuron = V1Neuron(shape=(5,), vth=10, tau=1, tau_exp=-3)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(lca_neuron.a_in)
        lca_neuron.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.run(condition=RunSteps(num_steps=5), run_cfg=run_config)

        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 246, -4, -1014],
                             [0, 3, 501, -17, -2037],
                             [0, 8, 756, -30, -3060]
                            ]).T
        
        actual = sign_extend_24bit(output_sink.data.get())

        input_source.stop()

        self.assertTrue(np.all(expected == actual), 
            f"Expected: {expected} Actual: {actual}")

    def test_reset_interval(self):
        input_data = np.array([[-1, 7, 256, -14, -1024]]).T
        input_source = source.RingBuffer(data=input_data)
        output_sink = sink.RingBuffer(shape=(5,), buffer=10)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        lca_neuron = V1Neuron(shape=(5,), vth=10, tau=1, tau_exp=-3,
                              reset_interval=4)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(lca_neuron.a_in)
        lca_neuron.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.run(condition=RunSteps(num_steps=10), run_cfg=run_config)

        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 246, -4, -1014],
                             [0, 3, 501, -17, -2037],
                             [0, 8, 756, -30, -3060],
                             [0, 0, 0, 0, 0],
                             [0, 0, 246, -4, -1014],
                             [0, 3, 501, -17, -2037],
                             [0, 8, 756, -30, -3060],
                             [0, 0, 0, 0, 0]
                            ]).T
        
        actual = sign_extend_24bit(output_sink.data.get())

        input_source.stop()

        self.assertTrue(np.all(expected == actual), 
            f"Expected: {expected} Actual: {actual}")

    def test_one_layer(self):
        input_data = np.array([[-1, 7, 256, -14, -1024]]).T
        bias = np.array([-2, 3, -4, 6, 7])

        input_source = source.RingBuffer(data=input_data)
        output_sink = sink.RingBuffer(shape=(5,), buffer=5)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        lca_neuron = V1Neuron(shape=(5,), vth=10, tau=1, tau_exp=-3,
                              two_layer=False, bias=bias)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(lca_neuron.a_in)
        lca_neuron.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.run(condition=RunSteps(num_steps=5), run_cfg=run_config)


        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 3, 238, 0, -1001],
                             [0, 11, 459, -1, -1892],
                             [0, 18, 652, -8, -2671]
                            ]).T
        
        actual = sign_extend_24bit(output_sink.data.get())

        input_source.stop()

        self.assertTrue(np.all(expected == actual), 
            f"Expected: {expected} Actual: {actual}")

    def test_reset_one_layer(self):
        input_data = np.array([[-1, 7, 256, -14, -1024]]).T
        bias = np.array([-2, 3, -4, 6, 7])

        input_source = source.RingBuffer(data=input_data)
        output_sink = sink.RingBuffer(shape=(5,), buffer=10)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        lca_neuron = V1Neuron(shape=(5,), vth=10, tau=1, tau_exp=-3,
                              two_layer=False, bias=bias, reset_interval=4)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(lca_neuron.a_in)
        lca_neuron.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.run(condition=RunSteps(num_steps=10), run_cfg=run_config)

        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 242, 0, -1007],
                             [0, 9, 463, -5, -1897],
                             [0, 17, 656, -11, -2676],
                             [0, 0, 0, 0, 0],
                             [0, 0, 242, 0, -1007],
                             [0, 9, 463, -5, -1897],
                             [0, 17, 656, -11, -2676],
                             [0, 0, 0, 0, 0]
                            ]).T
        
        actual = sign_extend_24bit(output_sink.data.get())

        input_source.stop()

        self.assertTrue(np.all(expected == actual), 
            f"Expected: {expected} Actual: {actual}")


if __name__ == "__main__":
    unittest.main()
