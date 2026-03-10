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


def momentum_monte_carlo(params:list, inds:list, ansatz:QuantumCircuit,
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
    print("Energy after MomentumBuilder and Monte Carlo: ", energy_final)
    
    return optimized_ansatz, optimized_params

def momentum_sa_merged(params:list, inds:list, ansatz:QuantumCircuit,
                         circuit:QuantumCircuit, hamiltonian:SparsePauliOp,
                         estimator:Estimator, beta1:float, beta2:float,
                         iters:int=2, optimization_runs:int=100):
    """
    Ansatz optimization pipeline that runs simulated annealing at 
    every layer as MomentumBuilder is running.
    """
    num_qubits = circuit.num_qubits
    M = np.zeros((len(params))) # Momentum
    currCirc = QuantumCircuit(num_qubits)
    currCirc = currCirc.compose(ansatz)

    for iter in range(iters):
        # Run simulated annealing to optimize params
        observables = [*hamiltonian.paulis, hamiltonian]
        simulator = AerSimulator(method='statevector')
        sa_params = MonteCarlo.simulated_annealing(
            optimization_runs, params, currCirc, simulator, observables, estimator
        )
        # print(sa_params)
        
        # Calculate momentum
        accumulator = []
        for i in range(len(sa_params)):
            # print(f"gradi = {abs(gradi(i, sa_params, currCirc, hamiltonian, estimator))}")
            # grad_i = abs(gradi(i, sa_params, currCirc, hamiltonian, estimator)[len(hamiltonian)-1]).item()
            grad_i = gradi(i, sa_params, currCirc, hamiltonian, estimator)
            M[i] = beta1 * M[i] + (1-beta1) * grad_i
            # print(f"M[i] = {M[i]}")
            # print(f"accumulator = {accumulator}")
            # print(f"inds[i] = {inds[i]}\n")
            heapq.heappush(accumulator, (M[i], inds[i]))

        # Construct momentum layer
        # print(f"accumulator after: {accumulator}")
        # print(f"num_qubits = {num_qubits}")
        mLayer, nparams, ninds = momen_layer(iter, num_qubits, accumulator)
        # print(f"sa_params = {sa_params}")
        # print(f"nparams = {nparams}")
        # print(f"inds = {inds}")
        # print(f"ninds = {ninds}")
        params = params + nparams
        inds = inds + ninds
        M = np.concatenate((M, len(nparams)*[0]))
        ansatz = ansatz.compose(mLayer)
        currCirc = circuit.compose(ansatz)

    circuit = circuit.compose(ansatz)
    cost_final = cost_func(params, circuit, observables, estimator)
    energy_final = cost_final[-1] # Last element in the list is the energy
    print("Energy after merged MB and SA: ", energy_final)
    
    return circuit


if __name__ == "__main__":
    H = SparsePauliOp.from_list([("ZIZZ", 1), ("ZZII", 3), ("IZZI", 1), ("IIZZ", 1)])
    
    angle1 = Parameter("angle1")
    angle2 = Parameter("angle2")
    angle3 = Parameter("angle3")
    angle4 = Parameter("angle4")
    
    circuit = QuantumCircuit(4)
    ansatz = QuantumCircuit(4)

    ansatz.rx(angle1, 0)
    ansatz.rx(angle2, 1)
    ansatz.rx(angle3, 2)
    ansatz.rx(angle4, 3)

    # ansatz.draw(output="mpl")

    # Run MomentumBuilder for comparison
    # observables = [*H.paulis,H]
    # final_circuit_MB = MomentumBuilder.MomentumBuilder([1,1,1,1], [0,1,2,3], ansatz, circuit, observables, Estimator(), 0.9, 0.99)
    # final_circuit_MB.draw(output="mpl")

    final_circuit_MMC, final_params = momentum_monte_carlo([1,1,1,1], [0,1,2,3], ansatz, circuit, H, Estimator(),
        beta1=0.9, beta2=0.99, iters=2, optimization_runs=100
    )
    final_circuit_MSA = momentum_sa_merged([1,1,1,1], [0,1,2,3], ansatz, circuit, H, Estimator(),
        beta1=0.9, beta2=0.99, iters=2, optimization_runs=100
    )
    final_circuit_MMC.draw(output="mpl")
    
    # print(f"Optimization complete. Final parameters: {final_params}")
    # plt.show()

