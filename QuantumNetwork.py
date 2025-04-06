import pennylane as qml
import torch
import torch.nn as nn

n_qubits = 4
q_depth = 6

dev = qml.device("default.qubit", wires=n_qubits)


def quantum_net(q_input_features, q_weights, design):
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)
    for layer in range(q_depth):
        for node in range(n_qubits):
            layer_node0 = design[str(layer % 6) + str(node % 4) + '0']
            layer_node1 = design[str(layer % 6) + str(node % 4) + '1']
            layer_node2 = design[str(layer % 6) + str(node % 4) + '2']
            if layer_node0:
                qml.RY(q_input_features[node], wires=node)
            if layer_node1 == 'x':
                qml.RX(q_weights[layer][node], wires=node)
            if layer_node1 == 'y':
                qml.RY(q_weights[layer][node], wires=node)
            if layer_node1 == 'z':
                qml.RZ(q_weights[layer][node], wires=node)
            if layer_node2 == 'H':
                qml.Hadamard(wires=node)
            if layer_node2 == 'Px':
                qml.PauliX(wires=node)
            if layer_node2 == 'Py':
                qml.PauliY(wires=node)
            if layer_node2 == 'Pz':
                qml.PauliZ(wires=node)
            if layer_node2 == 'CNot':
                qml.CNOT(wires=[node, (node + 1) % n_qubits])
            if layer_node2 == 'CSwap':
                qml.CSWAP(wires=[node, (node + 1) % n_qubits, (node + 2) % n_qubits])
            if layer_node2 == 'Tof':
                qml.Toffoli(wires=[node, (node + 1) % n_qubits, (node + 2) % n_qubits])
            if layer_node2 == 'CZ':
                qml.CZ(wires=[node, (node + 1) % n_qubits])


##############################################################################

@qml.qnode(dev, interface="torch")
def fidelity_qnode(x1, x2, q_weights_flat, design):
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    quantum_net(x1, q_weights, design)
    qml.adjoint(quantum_net)(x2, q_weights, design)

    return qml.probs(wires=range(n_qubits))


class QuantumLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.q_params = nn.Parameter(torch.randn(q_depth * n_qubits, dtype=torch.float64))

    def forward(self, x1, x2, design):
        x1 = x1.double()
        x2 = x2.double()

        fidelities = []
        for a, b in zip(x1, x2):
            probs = fidelity_qnode(a, b, self.q_params, design=design)
            fidelities.append(probs[0])

        return torch.stack(fidelities)


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.QuantumLayer = QuantumLayer()

    def forward(self, x1, x2, design):
        fidelity = self.QuantumLayer(x1, x2, design)
        return fidelity
