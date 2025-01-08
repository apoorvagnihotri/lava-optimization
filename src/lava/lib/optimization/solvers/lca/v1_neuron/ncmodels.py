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

import os

import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
from lava.magma.core.model.nc.var import NcVar
from lava.magma.core.model.nc.model import AbstractNcProcessModel
from lava.magma.core.model.nc.tables import Nodes
from lava.magma.core.model.nc.type import LavaNcType
from lava.magma.core.resources import Loihi2NeuroCore
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.solvers.lca.v1_neuron.process import \
    V1Neuron

try:
    from lava.magma.core.model.nc.net import NetL2
except ImportError:
    class NetL2:
        pass


@implements(proc=V1Neuron, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NcV1Neuron(AbstractNcProcessModel):
    a_in: NcInPort = LavaNcType(NcInPort, np.int32, precision=24)
    s_out: NcOutPort = LavaNcType(NcOutPort, np.int32, precision=24)
    v: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    bias: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    vth: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    tau: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    tau_exp: NcVar = LavaNcType(NcVar, np.int32, precision=24)
    reset_interval: NcVar = LavaNcType(NcVar, np.int32, precision=24)

    def allocate(self, net: NetL2):
        T = self.tau.var.get()
        T_exp = self.tau_exp.var.get()
        T_int = int(np.ldexp(T, T_exp + 24))

        shape = np.product(list(self.proc_params['shape']))

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        if self.proc_params['two_layer']:
            ucode_file = os.path.join(curr_dir, 'v1_neuron_2_layer.dasm')
        else:
            ucode_file = os.path.join(curr_dir, 'v1_neuron_1_layer.dasm')
        dv = 2**24 - T_int

        # Allocate program memory
        neurons_cfg: Nodes = net.neurons_cfg.allocate_ucode(
            shape=(1,),
            ucode=ucode_file,
            vth=self.vth.var.get(),
            tau=T_int,
            dv=dv,
            interval_sub_1=(self.reset_interval.var.get() - 1)
        )

        if self.proc_params['two_layer']:
            # Allocate individual neurons
            neurons: Nodes = net.neurons.allocate_ucode(
                shape=shape,
                v=self.v,
            )
        else:
            # Allocate individual neurons
            neurons: Nodes = net.neurons.allocate_ucode(
                shape=shape,
                v=self.v,
                bias=self.bias,
            )

        # Allocate output axons
        ax_out: Nodes = net.ax_out.allocate(shape=shape, num_message_bits=24)

        # Connect InPort of Process to neurons
        self.a_in.connect(neurons)

        # Connect Nodes
        neurons.connect(neurons_cfg)
        neurons.connect(ax_out)

        # Connect output axon to OutPort of Process
        ax_out.connect(self.s_out)