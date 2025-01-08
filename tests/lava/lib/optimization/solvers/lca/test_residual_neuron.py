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
from lava.magma.core.run_configs import Loihi2HwCfg

from lava.lib.optimization.solvers.lca.residual_neuron.process import \
    ResidualNeuron
from lava.lib.optimization.solvers.lca.util import sign_extend_24bit


class TestResidual(unittest.TestCase):
    def test_compilation(self):
        input_source = source.RingBuffer(data=np.array([[8, 7, 0, 8, 7]]).T)
        output_sink = sink.RingBuffer(shape=(5,), buffer=1)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        accumulator = ResidualNeuron(shape=(5,), spike_height=1, bias=0)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(accumulator.a_in)
        accumulator.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.compile(run_cfg=run_config)
        
        input_source.stop()

    def test_pos_threshold(self):
        input_source = source.RingBuffer(data=np.array([[4, 5, 10, 11, 40]]).T)
        output_sink = sink.RingBuffer(shape=(5,), buffer=1)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        accumulator = ResidualNeuron(shape=(5,), spike_height=10, bias=0)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(accumulator.a_in)
        accumulator.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.run(condition=RunSteps(num_steps=2), run_cfg=run_config)

        expected = np.array([0, 0, 10, 11, 40])
        actual = output_sink.data.get()[:, 0]

        input_source.stop()

        self.assertTrue(np.all(expected == actual), 
            f"Expected: {expected} Actual: {actual}")
        
        
    def test_neg_threshold(self):
        input_source = source.RingBuffer(
            data=np.array([[-4, -5, -10, -11, -40]]).T)
        output_sink = sink.RingBuffer(shape=(5,), buffer=1)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        accumulator = ResidualNeuron(shape=(5,), spike_height=10, bias=0)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(accumulator.a_in)
        accumulator.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.run(condition=RunSteps(num_steps=2), run_cfg=run_config)

        expected = np.array([0, 0, -10, -11, -40])
        actual = output_sink.data.get()[:, 0]

        input_source.stop()

        self.assertTrue(np.all(expected == actual), 
            f"Expected: {expected} Actual: {actual}")

    def test_accumulation_threshold(self):
        input_source = source.RingBuffer(data=np.array([[3, -5, -10, 5, -4]]).T)
        output_sink = sink.RingBuffer(shape=(5,), buffer=5)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        accumulator = ResidualNeuron(shape=(5,), spike_height=10, bias=0)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(accumulator.a_in)
        accumulator.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.run(condition=RunSteps(num_steps=5), run_cfg=run_config)

        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, -10, 0, 0],
                             [0, -10, -10, 10, 0],
                             [0, 0, -10, 0, -12],
                             [12, -10, -10, 10, 0]
                            ]).T
        
        actual = output_sink.data.get()

        input_source.stop()

        self.assertTrue(np.all(expected == actual), 
            f"Expected: {expected} Actual: {actual}")

    def test_bias(self):
        input_source = source.RingBuffer(data=np.array([[1, 0, 0, 1, 0]]).T)
        output_sink = sink.RingBuffer(shape=(5,), buffer=5)

        out_nx = NxToPyAdapter(shape=(5,), num_message_bits=24)
        in_nx = PyToNxAdapter(shape=(5,), num_message_bits=24)

        synapse = Dense(weights=np.eye(5), num_message_bits=24)

        bias = np.array([2, -5, -10, 4, -4])

        accumulator = ResidualNeuron(shape=(5,), spike_height=10, bias=bias)

        run_config = Loihi2HwCfg()

        input_source.s_out.connect(in_nx.inp)
        in_nx.out.connect(synapse.s_in)

        synapse.a_out.connect(accumulator.a_in)
        accumulator.s_out.connect(out_nx.inp)

        out_nx.out.connect(output_sink.a_in)

        input_source.run(condition=RunSteps(num_steps=5), run_cfg=run_config)

        expected = np.array([[0, 0, -10, 0, 0],
                             [0, -10, -10, 0, 0],
                             [0, 0, -10, 14, -12],
                             [11, -10, -10, 0, 0],
                             [0, 0, -10, 10, 0]
                            ]).T
        
        actual = sign_extend_24bit(output_sink.data.get())

        input_source.stop()

        self.assertTrue(np.all(expected == actual), 
            f"Expected: {expected} Actual: {actual}")

if __name__ == "__main__":
    unittest.main()