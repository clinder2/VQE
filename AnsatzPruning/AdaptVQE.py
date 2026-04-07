"""
AdaptVQE.py

Standalone ADAPT-VQE benchmark using the same 10-qubit Hamiltonian as the
MomentumBuilder + simulated annealing script.

This follows the IBM/Qiskit AdaptVQE pattern from the provided PDF:
    vqe = VQE(StatevectorEstimator(), QuantumCircuit(...), optimizer)
    adapt_vqe = AdaptVQE(vqe, operators=pool)
    result = adapt_vqe.compute_minimum_eigenvalue(hamiltonian)

The PDF also notes that AdaptVQE returns an AdaptVQEResult containing runtime
information such as number of iterations, termination criterion, and final
maximum gradient. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
from qiskit_algorithms.optimizers import SLSQP


def build_hamiltonian() -> SparsePauliOp:
    """Same Hamiltonian as in your __main__."""
    return SparsePauliOp.from_list(
        [
            ("ZZIZZZIIIZ", 1),
            ("IIZZIIZIZI", 1),
            ("ZIZIZIZZII", 1),
            ("IIIZZZIZZZ", 1),
            ("ZZIZZIZIII", 1),
            ("IIIZZIIIZZ", 1),
            ("IZZIZIZIIZ", 1),
            ("IZZZIZZIIZ", 1),
            ("ZIIIIZZIIZ", 1),
            ("IIZZIIZZZI", 1),
        ]
    )


def build_reference_ansatz() -> QuantumCircuit:
    """
    Matches the reference ansatz structure in your script:
    one RX parameter on each qubit.
    """
    qc = QuantumCircuit(10)
    angles = [Parameter(f"angle{i}") for i in range(1, 11)]
    for q, theta in enumerate(angles):
        qc.rx(theta, q)
    return qc


def build_operator_pool(num_qubits: int) -> list[SparsePauliOp]:
    """
    Operator pool for AdaptVQE.

    For this Hamiltonian (sum of Z-only strings), a pure Z pool is usually a poor
    benchmark because exp(-i * theta * Z) mostly adds phases and may not change
    the expectation meaningfully from simple reference states.

    A single-qubit Y pool is a simple, effective choice because e^{-i theta Y}
    rotates populations and can change expectations of Z-string Hamiltonians.
    """
    pool = []
    for pauli in ["X", "Y"]:
        for q in range(num_qubits):
            label = ["I"] * num_qubits
            label[q] = pauli
            pool.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return pool

def exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    """
    Since the Hamiltonian is diagonal in the computational basis (Z strings only),
    its exact ground energy can be obtained directly from the diagonal.
    """
    mat = hamiltonian.to_matrix(sparse=False)
    eigvals = np.linalg.eigvalsh(mat)
    return float(np.min(eigvals))


def run_adapt_vqe(
    hamiltonian: SparsePauliOp,
    initial_state: QuantumCircuit,
    operator_pool: list[SparsePauliOp],
    max_iterations: int = 25,
    gradient_threshold: float = 1e-5,
    eigenvalue_threshold: float = 1e-8,
):
    """
    Run IBM/Qiskit AdaptVQE using a trivial base VQE ansatz and an explicit pool.
    This is the API pattern described in the PDF. :contentReference[oaicite:2]{index=2}
    """
    num_qubits = hamiltonian.num_qubits

    # Base ansatz is a trivial circuit because AdaptVQE will build the ansatz
    # from the supplied operator pool.
    base_ansatz = QuantumCircuit(num_qubits)

    estimator = StatevectorEstimator()
    optimizer = SLSQP(maxiter=500)

    vqe = VQE(
        estimator=estimator,
        ansatz=base_ansatz,
        optimizer=optimizer,
    )

    adapt_vqe = AdaptVQE(
        solver=vqe,
        operators=operator_pool,
        gradient_threshold=gradient_threshold,
        eigenvalue_threshold=eigenvalue_threshold,
        max_iterations=max_iterations,
        initial_state=initial_state,
    )

    result = adapt_vqe.compute_minimum_eigenvalue(hamiltonian)
    return result


def safe_getattr(obj, name: str, default=None):
    return getattr(obj, name, default)


def main():
    H = build_hamiltonian()
    reference_rx_ansatz = build_reference_ansatz()
    operator_pool = build_operator_pool(H.num_qubits)

    print("=" * 80)
    print("ADAPT-VQE benchmark")
    print("=" * 80)
    print(f"Number of qubits: {H.num_qubits}")
    print(f"Reference RX ansatz parameter count (from your script): {reference_rx_ansatz.num_parameters}")
    print(f"ADAPT operator pool size: {len(operator_pool)}")
    print()

    exact_e0 = exact_ground_energy(H)
    print(f"Exact ground energy (classical diagonalization): {exact_e0}")
    print()

    reference_rx_ansatz = build_reference_ansatz()
    rx_initial_params = np.ones(reference_rx_ansatz.num_parameters)
    initial_state = reference_rx_ansatz.assign_parameters(rx_initial_params)

    result = run_adapt_vqe(
        hamiltonian=H,
        initial_state=initial_state,
        operator_pool=operator_pool,
        max_iterations=25,
        gradient_threshold=1e-8,
        eigenvalue_threshold=1e-10,
    )

    print("=" * 80)
    print("ADAPT-VQE result")
    print("=" * 80)

    eigenvalue = safe_getattr(result, "eigenvalue", None)
    optimal_value = safe_getattr(result, "optimal_value", None)
    final_energy = float(np.real(eigenvalue if eigenvalue is not None else optimal_value))
    

    print(f"Final ADAPT-VQE energy: {final_energy}")
    print(f"Energy error vs exact:   {final_energy - exact_e0}")
    print()

    # AdaptVQEResult-specific fields mentioned in the docs/PDF.
    print("ADAPT metadata:")
    print(f"  num_iterations:        {safe_getattr(result, 'num_iterations', 'N/A')}")
    print(f"  final_max_gradient:    {safe_getattr(result, 'final_max_gradient', 'N/A')}")
    print(f"  termination_criterion: {safe_getattr(result, 'termination_criterion', 'N/A')}")
    print()

    # VQE-like fields.
    print("Optimization metadata:")
    print(f"  optimal_point:         {safe_getattr(result, 'optimal_point', 'N/A')}")
    print(f"  optimal_parameters:    {safe_getattr(result, 'optimal_parameters', 'N/A')}")
    print(f"  cost_function_evals:   {safe_getattr(result, 'cost_function_evals', 'N/A')}")
    print(f"  optimizer_time:        {safe_getattr(result, 'optimizer_time', 'N/A')}")
    print()

    # Some versions expose a history.
    history = safe_getattr(result, "eigenvalue_history", None)
    if history is not None:
        print("Eigenvalue history:")
        for i, val in enumerate(history, start=1):
            print(f"  iter {i:2d}: {np.real(val)}")
        print()

    ansatz = safe_getattr(result, "optimal_circuit", None)
    if ansatz is None:
        ansatz = safe_getattr(result, "ansatz", None)

    if ansatz is not None:
        print("Final ADAPT ansatz:")
        print(ansatz.draw(output="text"))
        print(f"Final ansatz parameter count: {ansatz.num_parameters}")
    else:
        print("Final ADAPT ansatz: N/A")

    print()
    print("=" * 80)
    print("Notes")
    print("=" * 80)
    print(
        "This is a clean benchmark file for comparison against your MomentumBuilder "
        "+ simulated annealing pipeline on the same Hamiltonian."
    )
    print(
        "If you want a closer apples-to-apples comparison, you can also compare:\n"
        "  1) final energy\n"
        "  2) number of variational parameters\n"
        "  3) wall-clock runtime\n"
        "  4) number of optimizer evaluations"
    )


if __name__ == "__main__":
    main()