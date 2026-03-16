import sys
import os
import time
import heapq
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_aer.aerprovider import AerSimulator
from Optimization import MonteCarlo
from AnsatzPruning import MomentumBuilder
from AnsatzPruning.Utilities import cost_func
from qiskit.circuit import Parameter
from MomentumBuilder import momen_layer
from Utilities import *


def momentum_sa_phased(params:list, inds:list, ansatz:QuantumCircuit,
                         circuit:QuantumCircuit, hamiltonian:SparsePauliOp,
                         estimator:Estimator, beta1:float, beta2:float,
                         iters:int=2, optimization_runs:int=100):
    """
    Ansatz optimization pipeline that first runs MomentumBuilder and then optimizes 
    the parameters using Monte Carlo optimization method.
    """
    observables = [*hamiltonian.paulis, hamiltonian]
    
    # Run MomentumBuilder
    # print("Running MomentumBuilder")
    optimized_ansatz = MomentumBuilder.MomentumBuilder(
        params, inds, ansatz, circuit, observables, estimator,
        beta1, beta2, iters
    )
    
    # Extract parameters from ansatz
    num_params = len(optimized_ansatz.parameters)
    initial_params = np.ones(num_params)
    cost_mb = cost_func(initial_params, optimized_ansatz, observables, estimator)
    energy_mb = cost_mb[-1] # Last element in the list is the energy
    print("Energy after MomentumBuilder: ", energy_mb)
    
    # Run Monte Carlo optimization
    # print(f"Running Monte Carlo (stochastic hill climbing)")
    simulator = AerSimulator(method='statevector')
    initial_params = initial_params.copy()

    # Stochastic hill climbing
    # start_time = time.perf_counter()
    # optimized_params = MonteCarlo.stochastic_hill_climbing(
    #     optimization_runs, initial_params, optimized_ansatz, simulator, observables, estimator
    # )
    # end_time = time.perf_counter()
    # print(f"stochastic_hill_climbing took {end_time - start_time:.3f} seconds.")

    # Differential Evolution
    # start_time = time.perf_counter()
    # optimized_params = MonteCarlo.diff_evolution(
    #     optimization_runs, initial_params, optimized_ansatz, simulator, observables, estimator
    # )
    # end_time = time.perf_counter()
    # print(f"diff_evolution took {end_time - start_time:.3f} seconds.")

    # Global best PSO
    # start_time = time.perf_counter()
    # optimized_params = MonteCarlo.gbest_pso(
    #     optimization_runs, initial_params, optimized_ansatz, simulator, observables, estimator
    # )
    # end_time = time.perf_counter()
    # print(f"gbest_pso took {end_time - start_time:.3f} seconds.")

    # Simulated Annealing
    # start_time = time.perf_counter()
    optimized_params = MonteCarlo.simulated_annealing(
        optimization_runs, initial_params, optimized_ansatz, simulator, observables, estimator
    )
    # end_time = time.perf_counter()
    # print(f"simulated_annealing took {end_time - start_time:.3f} seconds.")


    cost_final = cost_func(optimized_params, optimized_ansatz, observables, estimator)
    energy_final = cost_final[-1] # Last element in the list is the energy
    print("Energy after MomentumBuilder and Simulated Annealing (SA), two-phased: ", energy_final)
    
    return optimized_ansatz, optimized_params

def momentum_sa_merged(params:list, inds:list, ansatz:QuantumCircuit,
                         circuit:QuantumCircuit, hamiltonian:SparsePauliOp,
                         estimator:Estimator, beta1:float, beta2:float,
                         iters:int=2, optimization_runs:int=100):
    """
    Ansatz optimization pipeline that constructs every layer using
    momentum and optimizes the layer's parameters using simulated annealing.
    """
    num_qubits = circuit.num_qubits
    observables = [*hamiltonian.paulis, hamiltonian]
    M = np.zeros((len(params))) # Momentum
    currCirc = QuantumCircuit(num_qubits)
    currCirc = currCirc.compose(ansatz)

    for iter in range(iters):
        # Calculate momentum
        accumulator = []
        for i in range(len(params)):
            grad_i = abs(gradi(i, params, currCirc, hamiltonian, estimator)).item()
            M[i] = beta1 * M[i] + (1-beta1) * grad_i
            heapq.heappush(accumulator, (-M[i], inds[i]))

        # Construct momentum layer and append it to circuit
        keep = max(2, num_qubits // 2) # keep = how many qubits with highest momentums we use
        momentum_layer, new_params, new_inds = momen_layer(iter, num_qubits, accumulator, keep=keep)
        params = params + new_params
        inds = inds + new_inds
        M = np.concatenate((M, len(new_params)*[0]))
        ansatz = ansatz.compose(momentum_layer)
        currCirc = circuit.compose(ansatz)

        # Run simulated annealing to optimize params
        simulator = AerSimulator(method='statevector')
        sa_params = MonteCarlo.simulated_annealing(
            optimization_runs, np.array(params), currCirc, simulator, observables, estimator
        )
        params = list(sa_params)
        # print(sa_params)

    circuit = circuit.compose(ansatz)
    cost_final = cost_func(params, circuit, observables, estimator)
    energy_final = cost_final[-1] # Last element in the list is the energy
    print("Energy after merged MB and SA: ", energy_final)
    
    return circuit


if __name__ == "__main__":
    # 10 qubits:
    H = SparsePauliOp.from_list([("ZZIZZZIIIZ", 1), ("IIZZIIZIZI", 1), ("ZIZIZIZZII", 1), ("IIIZZZIZZZ", 1), 
                                 ("ZZIZZIZIII", 1), ("IIIZZIIIZZ", 1), ("IZZIZIZIIZ", 1), ("IZZZIZZIIZ", 1), 
                                 ("ZIIIIZZIIZ", 1), ("IIZZIIZZZI", 1)])
    
    angle1 = Parameter("angle1")
    angle2 = Parameter("angle2")
    angle3 = Parameter("angle3")
    angle4 = Parameter("angle4")
    angle5 = Parameter("angle5")
    angle6 = Parameter("angle6")
    angle7 = Parameter("angle7")
    angle8 = Parameter("angle8")
    angle9 = Parameter("angle9")
    angle10 = Parameter("angle10")
    
    circuit = QuantumCircuit(10)
    ansatz = QuantumCircuit(10)

    ansatz.rx(angle1, 0)
    ansatz.rx(angle2, 1)
    ansatz.rx(angle3, 2)
    ansatz.rx(angle4, 3)
    ansatz.rx(angle5, 4)
    ansatz.rx(angle6, 5)
    ansatz.rx(angle7, 6)
    ansatz.rx(angle8, 7)
    ansatz.rx(angle9, 8)
    ansatz.rx(angle10, 9)

    # ansatz.draw(output="mpl")

    # Run MomentumBuilder for comparison
    observables = [*H.paulis,H]
    final_circuit_MB = MomentumBuilder.MomentumBuilder([1,1,1,1,1,1,1,1,1,1], [0,1,2,3,4,5,6,7,8,9], ansatz, circuit, observables, Estimator(), 0.9, 0.99)
    final_circuit_MB.draw(output="mpl")

    final_circuit_MMC, final_params = momentum_sa_phased([1,1,1,1,1,1,1,1,1,1], [0,1,2,3,4,5,6,7,8,9], ansatz, circuit, H, Estimator(),
        beta1=0.9, beta2=0.99, iters=2, optimization_runs=100
    )
    final_circuit_MMC.draw(output="mpl")

    final_circuit_MSA = momentum_sa_merged([1,1,1,1,1,1,1,1,1,1], [0,1,2,3,4,5,6,7,8,9], ansatz, circuit, H, Estimator(),
        beta1=0.9, beta2=0.99, iters=2, optimization_runs=100
    )
    final_circuit_MSA.draw(output="mpl")
    
    # print(f"Optimization complete. Final parameters: {final_params}")
    plt.show()

