"""
Utilities for running ADAPT-VQE benchmarks with Qiskit.

This module follows the documented Qiskit Algorithms pattern:

    vqe = VQE(StatevectorEstimator(), QuantumCircuit(num_qubits), optimizer)
    adapt_vqe = AdaptVQE(vqe, operators=pool, initial_state=reference_state)
    result = adapt_vqe.compute_minimum_eigenvalue(hamiltonian)

When ``operators`` is supplied, ADAPT-VQE builds the evolved-operator ansatz
internally from that pool. The placeholder ansatz passed to ``VQE`` is only
used to satisfy the solver interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

try:
    from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
    from qiskit_algorithms.optimizers import SLSQP
except ImportError as exc:  # pragma: no cover - depends on local environment
    AdaptVQE = None  # type: ignore[assignment]
    VQE = None  # type: ignore[assignment]
    SLSQP = None  # type: ignore[assignment]
    _QISKIT_ALGORITHMS_IMPORT_ERROR = exc
else:
    _QISKIT_ALGORITHMS_IMPORT_ERROR = None


def ensure_qiskit_algorithms_available() -> None:
    """Raise a clear error if the standalone qiskit-algorithms package is missing."""
    if _QISKIT_ALGORITHMS_IMPORT_ERROR is not None:
        raise ImportError(
            "ADAPT-VQE requires the standalone `qiskit-algorithms` package. "
            "Install it with `pip install qiskit-algorithms`."
        ) from _QISKIT_ALGORITHMS_IMPORT_ERROR


def build_hamiltonian() -> SparsePauliOp:
    """Return the 10-qubit demo Hamiltonian used by this module's CLI example."""
    return SparsePauliOp.from_list(
        [
            ("ZZIZZZIIIZ", 1.0),
            ("IIZZIIZIZI", 1.0),
            ("ZIZIZIZZII", 1.0),
            ("IIIZZZIZZZ", 1.0),
            ("ZZIZZIZIII", 1.0),
            ("IIIZZIIIZZ", 1.0),
            ("IZZIZIZIIZ", 1.0),
            ("IZZZIZZIIZ", 1.0),
            ("ZIIIIZZIIZ", 1.0),
            ("IIZZIIZZZI", 1.0),
        ]
    )


def build_reference_ansatz(num_qubits: int = 10) -> QuantumCircuit:
    """Return a parameterized RX reference ansatz with one angle per qubit."""
    qc = QuantumCircuit(num_qubits)
    angles = [Parameter(f"angle{i}") for i in range(num_qubits)]
    for qubit, theta in enumerate(angles):
        qc.rx(theta, qubit)
    return qc


def build_reference_state(num_qubits: int, angle: float = 1.0) -> QuantumCircuit:
    """
    Return a parameter-free RX product state.

    This mirrors the Set Cover benchmark's initial parameter choice of all ones.
    """
    reference = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        reference.rx(angle, qubit)
    return reference


def build_operator_pool(
    num_qubits: int,
    paulis: Sequence[str] = ("X", "Y"),
) -> list[SparsePauliOp]:
    """
    Build a simple single-qubit operator pool for ADAPT-VQE.

    For the diagonal Z-string Hamiltonians used in this repository, a pure Z pool
    is not useful because it largely only changes phases. Single-qubit X/Y
    generators can change the measurement probabilities of computational-basis
    states and are a reasonable generic pool for these benchmarks.
    """
    pool: list[SparsePauliOp] = []
    for pauli in paulis:
        if len(pauli) != 1:
            raise ValueError(f"Expected single-qubit Pauli labels, got {pauli!r}.")
        for qubit in range(num_qubits):
            label = ["I"] * num_qubits
            label[qubit] = pauli
            pool.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return pool


def exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    """Return the exact ground-state energy from dense diagonalization."""
    matrix = hamiltonian.to_matrix(sparse=False)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return float(np.min(np.real_if_close(eigenvalues)))


def extract_result_metrics(result: Any) -> dict[str, Any]:
    """Extract a small, benchmark-friendly metrics dictionary from an ADAPT result."""
    eigenvalue = getattr(result, "eigenvalue", None)
    optimal_value = getattr(result, "optimal_value", None)
    energy = float(np.real(eigenvalue if eigenvalue is not None else optimal_value))

    optimal_circuit = getattr(result, "optimal_circuit", None)
    termination_criterion = getattr(result, "termination_criterion", None)
    termination_text = getattr(termination_criterion, "value", termination_criterion)

    history = getattr(result, "eigenvalue_history", None)
    if history is not None:
        history = [float(np.real_if_close(value)) for value in history]

    return {
        "energy": energy,
        "num_iterations": getattr(result, "num_iterations", None),
        "final_max_gradient": getattr(result, "final_max_gradient", None),
        "termination_criterion": termination_text,
        "cost_function_evals": getattr(result, "cost_function_evals", None),
        "optimizer_time": getattr(result, "optimizer_time", None),
        "optimal_point": getattr(result, "optimal_point", None),
        "optimal_parameters": getattr(result, "optimal_parameters", None),
        "num_parameters": (
            optimal_circuit.num_parameters if optimal_circuit is not None else None
        ),
        "ansatz_depth": optimal_circuit.depth() if optimal_circuit is not None else None,
        "eigenvalue_history": history,
        "optimal_circuit": optimal_circuit,
    }


def run_adapt_vqe(
    hamiltonian: SparsePauliOp,
    initial_state: QuantumCircuit | None = None,
    operator_pool: Sequence[SparsePauliOp] | None = None,
    max_iterations: int = 25,
    gradient_threshold: float = 1e-5,
    eigenvalue_threshold: float = 1e-8,
    optimizer_maxiter: int = 500,
):
    """
    Run Qiskit's ADAPT-VQE against a Hamiltonian using an explicit operator pool.

    Args:
        hamiltonian: target Hamiltonian.
        initial_state: optional parameter-free circuit prepended to the ADAPT ansatz.
        operator_pool: explicit operator pool. If omitted, a single-qubit X/Y pool
            is generated automatically.
        max_iterations: maximum ADAPT growth iterations.
        gradient_threshold: ADAPT convergence threshold on the maximum gradient.
        eigenvalue_threshold: ADAPT convergence threshold on eigenvalue change.
        optimizer_maxiter: maximum iterations for the inner SLSQP solve.
    """
    ensure_qiskit_algorithms_available()

    num_qubits = hamiltonian.num_qubits
    if initial_state is None:
        initial_state = build_reference_state(num_qubits)
    if operator_pool is None:
        operator_pool = build_operator_pool(num_qubits)

    estimator = StatevectorEstimator()
    optimizer = SLSQP(maxiter=optimizer_maxiter)

    # The solver ansatz is intentionally trivial because ADAPT-VQE constructs
    # the evolved-operator ansatz itself when `operators` is provided.
    solver = VQE(
        estimator=estimator,
        ansatz=QuantumCircuit(num_qubits),
        optimizer=optimizer,
    )

    adapt_vqe = AdaptVQE(
        solver=solver,
        operators=list(operator_pool),
        gradient_threshold=gradient_threshold,
        eigenvalue_threshold=eigenvalue_threshold,
        max_iterations=max_iterations,
        initial_state=initial_state,
    )

    return adapt_vqe.compute_minimum_eigenvalue(hamiltonian)


def main() -> None:
    """Run a small command-line demo on the built-in 10-qubit Hamiltonian."""
    hamiltonian = build_hamiltonian()
    exact_energy = exact_ground_energy(hamiltonian)

    result = run_adapt_vqe(
        hamiltonian=hamiltonian,
        initial_state=build_reference_state(hamiltonian.num_qubits, angle=1.0),
        operator_pool=build_operator_pool(hamiltonian.num_qubits),
        max_iterations=25,
        gradient_threshold=1e-8,
        eigenvalue_threshold=1e-10,
    )
    metrics = extract_result_metrics(result)

    print("=" * 80)
    print("ADAPT-VQE benchmark")
    print("=" * 80)
    print(f"Number of qubits: {hamiltonian.num_qubits}")
    print(f"Operator pool size: {len(build_operator_pool(hamiltonian.num_qubits))}")
    print(f"Exact ground energy: {exact_energy}")
    print(f"Final ADAPT-VQE energy: {metrics['energy']}")
    print(f"Energy error vs exact: {metrics['energy'] - exact_energy}")
    print()
    print("ADAPT metadata:")
    print(f"  num_iterations:        {metrics['num_iterations']}")
    print(f"  final_max_gradient:    {metrics['final_max_gradient']}")
    print(f"  termination_criterion: {metrics['termination_criterion']}")
    print(f"  cost_function_evals:   {metrics['cost_function_evals']}")
    print(f"  optimizer_time:        {metrics['optimizer_time']}")
    print(f"  num_parameters:        {metrics['num_parameters']}")
    print(f"  ansatz_depth:          {metrics['ansatz_depth']}")

    optimal_circuit = metrics["optimal_circuit"]
    if optimal_circuit is not None:
        print()
        print("Final ADAPT ansatz:")
        print(optimal_circuit.draw(output="text"))


if __name__ == "__main__":
    main()
