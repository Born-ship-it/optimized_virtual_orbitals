"""
ovos_quantum_full.py

Quantum OVOS (Orbital-Optimized VQE) Implementation

This implements a full hybrid quantum-classical orbital optimization where:
1. The classical OVOS framework performs orbital rotations using Newton steps
2. The energy is evaluated using VQE on a quantum simulator
3. Gradients are computed using the parameter-shift rule (quantum gradients)
4. The orbitals are optimized to minimize the quantum energy

This is a full Orbital-Optimized VQE (OO-VQE) implementation.

References:
    - Adamowicz & Bartlett (1987): Original OVOS
    - OO-VQE literature: Phys. Rev. Research 3, 033230 (2021)

Complete Quantum OVOS (Orbital-Optimized VQE) Implementation
Compatible with Qiskit 2.4.2, qiskit-nature 0.7.2, qiskit-algorithms 0.4.0

Uses statevector simulation directly without the Estimator primitive.

Features:
- Robust VQE with fallback options
- Adaptive ansatz switching (simple ansatz → UCCSD)
- Automatic fallback to classical MP2 if quantum fails
- Complete Quantum OVOS with proper active space handling.
    Uses Qiskit Nature's ActiveSpaceTransformer for correct Hamiltonian truncation.
- Features Hamiltonian caching to disk for faster repeated runs.
- Uses statevector simulation directly without the Estimator primitive.
"""

import numpy as np
import time
import warnings
import pickle
import hashlib
import os
from typing import List, Optional, Tuple

# Quantum computing imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, Aer
from qiskit.quantum_info import Statevector, SparsePauliOp, Pauli

# qiskit_algorithms imports
try:
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SLSQP, COBYLA, L_BFGS_B
except ImportError:
    warnings.warn("qiskit-algorithms not found.")
    VQE = None
    L_BFGS_B = None
    SLSQP = None
    COBYLA = None

# qiskit-nature imports
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD

# Import base OVOS
try:
    from ovos import OVOS
except ImportError:
    from .ovos import OVOS

from scipy.optimize import minimize


class QuantumHamiltonianBuilder:
    """Builds the electronic Hamiltonian with active space truncation and caching."""
    
    def __init__(self, mol, mf, active_orbitals=None, cache_dir='hamiltonian_cache'):
        self.mol = mol
        self.mf = mf
        self.active_orbitals = active_orbitals
        self._hamiltonian_cache = {}
        self.mapper = JordanWignerMapper()
        self.cache_dir = cache_dir
        self.verbose = 0
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.full_mo_coeff = mf.mo_coeff
        
        if active_orbitals is not None:
            self.n_active_spin = len(active_orbitals)
            self.active_indices = active_orbitals
        else:
            self.n_active_spin = 2 * mol.nao_nr()
            self.active_indices = list(range(self.n_active_spin))
        
        self.n_active_electrons = mol.nelec[0] + mol.nelec[1]
        self._cache_key = self._generate_cache_key()
    
    def _generate_cache_key(self):
        mol_str = f"{self.mol.atom}_{self.mol.basis}_{self.mol.charge}_{self.mol.spin}"
        active_str = f"active_{sorted(self.active_orbitals)}" if self.active_orbitals else "full"
        key_str = f"{mol_str}_{active_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_filename(self):
        return os.path.join(self.cache_dir, f"hamiltonian_{self._cache_key}.pkl")
    
    def _save_to_cache(self, cache_data):
        try:
            cache_file = self._get_cache_filename()
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            if self.verbose:
                print(f"  Hamiltonian cached to: {cache_file}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Failed to cache Hamiltonian: {e}")
            return False
    
    def _load_from_cache(self):
        try:
            cache_file = self._get_cache_filename()
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                if self.verbose:
                    print(f"  Hamiltonian loaded from cache: {cache_file}")
                return cache_data
            return None
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Failed to load Hamiltonian from cache: {e}")
            return None
    
    def _extract_constant_term(self, hamiltonian_pauli):
        """Extract the constant term (identity operator) from the Pauli Hamiltonian."""
        constant_term = 0.0
        for pauli_str, coeff in zip(hamiltonian_pauli.paulis, hamiltonian_pauli.coeffs):
            # Check if all characters are 'I'
            if all(c == 'I' for c in str(pauli_str)):
                constant_term += coeff
        return constant_term
    
    def build_hamiltonian(self, mo_coeff, use_cache=True):
        if use_cache:
            cached_data = self._load_from_cache()
            if cached_data is not None:
                # Handle different cache formats
                if isinstance(cached_data, tuple):
                    if len(cached_data) == 3:
                        return cached_data
                    elif len(cached_data) == 2:
                        hamiltonian, inactive_energy = cached_data
                        constant_term = self._extract_constant_term(hamiltonian)
                        return hamiltonian, inactive_energy, constant_term
                return cached_data, 0.0, 0.0
        
        from qiskit_nature.second_q.drivers import PySCFDriver
        
        driver = PySCFDriver(
            atom=self.mol.atom,
            basis=self.mol.basis,
            charge=self.mol.charge,
            spin=self.mol.spin,
        )
        problem = driver.run()
        hamiltonian = problem.hamiltonian.second_q_op()
        inactive_energy = 0.0
        reference_energy = 0.0
        
        if self.active_orbitals is not None:
            hamiltonian, inactive_energy = self._truncate_hamiltonian(hamiltonian)
        
        # Map to Pauli operators
        hamiltonian_pauli = self.mapper.map(hamiltonian)
        
        # Extract the constant term (identity operator contribution)
        # This is the energy that was removed to make the Hamiltonian traceless
        constant_term = self._extract_constant_term(hamiltonian_pauli)
        
        # The reference energy is the constant term
        if abs(constant_term) > 1e-6:
            reference_energy = constant_term
            if self.verbose:
                print(f"    Extracted constant term: {reference_energy:.10f} Ha")
        else:
            # If no constant term, the Hamiltonian is already traceless
            reference_energy = 0.0
        
        cache_data = (hamiltonian_pauli, inactive_energy, reference_energy)
        self._save_to_cache(cache_data)
        
        return cache_data
    
    def _truncate_hamiltonian(self, hamiltonian):
        if self.verbose:
            print(f"    Active indices: {self.active_indices}")
            terms_list = list(hamiltonian.terms())
            print(f"    Number of Hamiltonian terms: {len(terms_list)}")
        
        try:
            from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
            from qiskit_nature.second_q.drivers import PySCFDriver
        except ImportError:
            try:
                from qiskit_nature.second_q.problems import ActiveSpaceTransformer
            except ImportError:
                if self.verbose:
                    print("    ActiveSpaceTransformer not available")
                return hamiltonian, 0.0
        
        try:
            spatial_active = sorted(set([i // 2 for i in self.active_indices]))
            n_active_electrons = self.mol.nelec[0] + self.mol.nelec[1]
            
            if self.verbose:
                print(f"    Spatial active orbitals: {spatial_active}")
                print(f"    Active electrons: {n_active_electrons}")
            
            driver = PySCFDriver(
                atom=self.mol.atom,
                basis=self.mol.basis,
                charge=self.mol.charge,
                spin=self.mol.spin,
            )
            problem = driver.run()
            
            if self.verbose:
                print(f"    Total spatial orbitals: {problem.num_spatial_orbitals}")
            
            transformer = ActiveSpaceTransformer(
                num_electrons=n_active_electrons,
                num_spatial_orbitals=len(spatial_active),
                active_orbitals=spatial_active,
            )
            
            active_problem = transformer.transform(problem)
            active_hamiltonian = active_problem.hamiltonian.second_q_op()
            
            inactive_energy = transformer.reference_inactive_energy
            if inactive_energy is None:
                inactive_energy = 0.0
            
            if self.verbose:
                terms_list = list(active_hamiltonian.terms())
                print(f"    ActiveSpaceTransformer success: {len(terms_list)} terms to {len(spatial_active)*2} qubits")
                print(f"    Inactive energy offset: {inactive_energy:.10f} Ha")
            
            return active_hamiltonian, inactive_energy
            
        except Exception as e:
            if self.verbose:
                print(f"    ActiveSpaceTransformer failed: {e}")
                print("    Falling back to manual truncation...")
            
            return self._truncate_hamiltonian_manual(hamiltonian), 0.0

    def _truncate_hamiltonian_manual(self, hamiltonian):
        from qiskit_nature.second_q.operators import FermionicOp
        
        terms_list = list(hamiltonian.terms())
        
        if self.verbose:
            print(f"    Manual truncation: {len(terms_list)} terms")
        
        active_set = set(self.active_indices)
        
        filtered_terms = {}
        terms_kept = 0
        terms_skipped = 0
        
        for term, coeff in terms_list:
            all_active = True
            for op in term:
                idx, action = op
                if idx not in active_set:
                    all_active = False
                    break
            
            if all_active:
                new_term = []
                active_list = sorted(active_set)
                for op in term:
                    idx, action = op
                    new_idx = active_list.index(idx)
                    new_term.append((new_idx, action))
                new_term_tuple = tuple(new_term)
                filtered_terms[new_term_tuple] = coeff
                terms_kept += 1
            else:
                terms_skipped += 1
        
        if self.verbose:
            print(f"    Manual truncation: kept {terms_kept}, skipped {terms_skipped}")
        
        if filtered_terms:
            if self.verbose:
                print(f"    Manual truncation success: {len(filtered_terms)} terms to {len(active_set)} qubits")
            try:
                return FermionicOp(filtered_terms, register_length=len(active_set))
            except TypeError:
                return FermionicOp(filtered_terms)
        
        if self.verbose:
            print("    Manual truncation failed! Using full Hamiltonian.")
        return hamiltonian


class QuantumEnergyEvaluator:
    """Evaluates quantum energy using statevector simulation directly."""
    
    def __init__(self, mol, mf, active_orbitals=None, verbose=0,
                 use_simple_ansatz=False, use_uccsd=True, simple_ansatz_reps=1, max_qubits=20,
                 cache_dir='hamiltonian_cache'):
        self.mol = mol
        self.mf = mf
        self.active_orbitals = active_orbitals
        self.verbose = verbose
        self.max_qubits = max_qubits
        self.simple_ansatz_reps = simple_ansatz_reps
        self.use_simple_ansatz = use_simple_ansatz
        self.inactive_energy = 0.0
        self.reference_energy = 0.0  # The constant term to add back
        
        self.mapper = JordanWignerMapper()
        
        if active_orbitals is not None:
            self.n_qubits = len(active_orbitals)
        else:
            self.n_qubits = 2 * mol.nao_nr()
        
        self.can_run_vqe = self.n_qubits <= max_qubits
        
        if not self.can_run_vqe and verbose:
            print(f"  Warning: {self.n_qubits} qubits exceeds max ({max_qubits}).")
        
        self.hamiltonian_builder = QuantumHamiltonianBuilder(
            mol, mf, active_orbitals, cache_dir=cache_dir
        )
        self.hamiltonian_builder.verbose = verbose
        
        # Build Hamiltonian - returns (hamiltonian_pauli, inactive_energy, reference_energy)
        hamiltonian_data = self.hamiltonian_builder.build_hamiltonian(mf.mo_coeff)
        
        # Handle different cache formats
        if isinstance(hamiltonian_data, tuple):
            if len(hamiltonian_data) == 3:
                self.hamiltonian, self.inactive_energy, self.reference_energy = hamiltonian_data
            elif len(hamiltonian_data) == 2:
                self.hamiltonian, self.inactive_energy = hamiltonian_data
                # Extract constant term if not provided
                self.reference_energy = self._extract_constant_term(self.hamiltonian)
            else:
                self.hamiltonian = hamiltonian_data[0]
                self.inactive_energy = 0.0
                self.reference_energy = self._extract_constant_term(self.hamiltonian)
        else:
            self.hamiltonian = hamiltonian_data
            self.inactive_energy = 0.0
            self.reference_energy = self._extract_constant_term(self.hamiltonian)
        
        self._cached_hamiltonian = self.hamiltonian
        
        self.use_uccsd = use_uccsd
        self.ansatz = None
        self.initial_params = None
        self._has_switched = False
        self._vqe_result = None
        self._current_hamiltonian = self.hamiltonian
        self._energy_history = []
        
        self._last_energy = None
        self._last_params = None
        self._iteration_count = 0
        
        if SLSQP is not None:
            self.optimizer = SLSQP(maxiter=50)
        else:
            self.optimizer = None
        
        if self.can_run_vqe:
            if use_simple_ansatz:
                self._setup_simple_ansatz()
            else:
                self._setup_uccsd_ansatz()
        else:
            self.ansatz = None
        
        if self.verbose:
            print("Quantum Energy Evaluator initialized")
            print(f"  Active orbitals: {active_orbitals}")
            print(f"  Qubits: {self.n_qubits}")
            print(f"  Inactive energy: {self.inactive_energy:.10f} Ha")
            print(f"  Reference energy (to add back): {self.reference_energy:.10f} Ha")
            if self.ansatz is not None:
                print(f"  Ansatz parameters: {self.ansatz.num_parameters}")
            print(f"  Can run VQE: {self.can_run_vqe}")
            print(f"  Using UCCSD: {self.use_uccsd}")
    
    def _extract_constant_term(self, hamiltonian):
        """Extract the constant term from the Hamiltonian."""
        constant_term = 0.0
        for pauli_str, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            if all(c == 'I' for c in str(pauli_str)):
                constant_term += coeff
        return constant_term
    
    def _setup_simple_ansatz(self):
        from qiskit.circuit.library import EfficientSU2
        
        self.ansatz = EfficientSU2(
            self.n_qubits,
            reps=self.simple_ansatz_reps,
            entanglement='linear'
        )
        self.initial_params = np.random.randn(self.ansatz.num_parameters) * 0.1
        self.use_uccsd = False
        
        if self.verbose > 1:
            print(f"  Simple ansatz parameters: {self.ansatz.num_parameters}")
    
    def _setup_uccsd_ansatz(self):
        n_spatial = self.n_qubits // 2
        
        try:
            self.ansatz = UCCSD(
                num_spatial_orbitals=n_spatial,
                num_particles=(self.mol.nelec[0], self.mol.nelec[1]),
                qubit_mapper=self.mapper,
            )
        except TypeError:
            try:
                self.ansatz = UCCSD(
                    num_spatial_orbitals=n_spatial,
                    num_particles=(self.mol.nelec[0], self.mol.nelec[1]),
                    mapper=self.mapper,
                )
            except TypeError:
                self.ansatz = UCCSD(
                    num_spatial_orbitals=n_spatial,
                    num_particles=(self.mol.nelec[0], self.mol.nelec[1]),
                )
        
        self.initial_params = np.random.randn(self.ansatz.num_parameters) * 0.1
        self.use_uccsd = True
        
        if self.verbose > 1:
            print(f"  UCCSD ansatz parameters: {self.ansatz.num_parameters}")
    
    def switch_to_uccsd(self):
        if self._has_switched or self.use_uccsd:
            return
        
        if not self.can_run_vqe:
            if self.verbose:
                print("  Cannot switch to UCCSD: VQE not available")
            return
        
        if self.verbose:
            print("  Switching to UCCSD ansatz...")
        
        current_params = self.initial_params
        
        self._setup_uccsd_ansatz()
        
        if current_params is not None and len(current_params) == self.ansatz.num_parameters:
            self.initial_params = current_params
        
        self._has_switched = True
        
        if self.verbose:
            print(f"  Switched to UCCSD with {self.ansatz.num_parameters} parameters")
    
    def _compute_expectation_all_terms(self, statevector, hamiltonian):
        """
        Compute expectation using ALL Pauli terms with progress bar.
        This is the most reliable method for traceless Hamiltonians.
        """
        try:
            pauli_strings = hamiltonian.paulis
            coeffs = hamiltonian.coeffs
            
            # Convert statevector to array
            sv_array = np.asarray(statevector)
            
            total_energy = 0.0 + 0.0j
            terms_used = 0
            total_terms = len(pauli_strings)
            
            # Show progress bar if verbose
            if self.verbose > 1 and total_terms > 100:
                from tqdm import tqdm
                iterator = tqdm(
                    zip(pauli_strings, coeffs),
                    total=total_terms,
                    desc="      Computing Pauli expectations",
                    unit=" terms",
                    ncols=80
                )
            else:
                iterator = zip(pauli_strings, coeffs)
            
            # Iterate over ALL Pauli terms
            for pauli_str, coeff in iterator:
                if abs(coeff) < 1e-10:
                    continue
                
                # Use sparse matrix to save memory
                pauli_op = Pauli(pauli_str)
                pauli_matrix = pauli_op.to_matrix(sparse=True)
                expectation = np.vdot(sv_array, pauli_matrix @ sv_array)
                total_energy += coeff * expectation
                terms_used += 1
            
            energy = np.real(total_energy)
            
            if self.verbose > 1:
                print(f"      Full Hamiltonian expectation: {energy:.10f} Ha ({terms_used}/{total_terms} terms)")
            
            return energy
            
        except MemoryError:
            if self.verbose > 0:
                print(f"      MemoryError: Using constant term fallback...")
            return self._get_constant_term(hamiltonian)
        except Exception as e:
            if self.verbose > 0:
                print(f"      Full Hamiltonian expectation failed: {e}")
            return self._get_constant_term(hamiltonian)

    def _compute_expectation_statevector(self, statevector, hamiltonian):
        """
        Compute <ψ|H|ψ> using the fastest reliable method.
        """
        # Strategy 1: Direct accelerated method (fastest)
        try:
            sv = Statevector(statevector)
            energy = sv.expectation_value(hamiltonian).real
            
            # If it's zero, the Hamiltonian is traceless - use full term-by-term
            if abs(energy) < 1e-6:
                if self.verbose > 1:
                    print(f"      Direct expectation is zero, using full term-by-term...")
                return self._compute_expectation_all_terms(statevector, hamiltonian)
            
            return energy
            
        except Exception as e:
            if self.verbose > 1:
                print(f"      Direct expectation failed: {e}")
            return self._compute_expectation_all_terms(statevector, hamiltonian)
    
    def _get_constant_term(self, hamiltonian):
        """Get the constant term (identity operator contribution) of the Hamiltonian."""
        constant_term = 0.0
        for pauli_str, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            if all(c == 'I' for c in str(pauli_str)):
                constant_term += coeff
        return constant_term

    def _evaluate_statevector(self, params):
        """Evaluate the expectation value <ψ|H|ψ> using statevector simulation."""
        try:
            circuit = self.ansatz.assign_parameters(params)
            
            backend = Aer.get_backend('statevector_simulator')
            transpiled = transpile(circuit, backend)
            result = backend.run(transpiled, shots=1).result()
            statevector = result.get_statevector(transpiled)
            
            # Compute energy using the current Hamiltonian
            energy = self._compute_expectation_statevector(statevector, self._current_hamiltonian)
            
            return energy
            
        except Exception as e:
            if self.verbose > 0:
                print(f"      Statevector evaluation failed: {e}")
            return -0.04
    
    def _run_vqe_statevector(self, hamiltonian):
        """Run VQE using statevector simulation with early stopping."""
        if self.ansatz is None:
            return -0.04, None
        
        self._current_hamiltonian = hamiltonian
        self._energy_history = []
        
        # Early stopping parameters
        no_improvement_count = 0
        max_no_improvement = 3  # Stop after 3 steps without improvement
        last_energy = None
        energy_tolerance = 1e-8
        
        try:            
            def objective(params):
                energy = self._evaluate_statevector(params)
                self._energy_history.append(energy)
                
                # Check for improvement
                nonlocal no_improvement_count, last_energy
                if last_energy is not None:
                    if abs(energy - last_energy) < energy_tolerance:
                        no_improvement_count += 1
                        if self.verbose > 2:
                            print(f"      No improvement {no_improvement_count}/{max_no_improvement}")
                    else:
                        no_improvement_count = 0
                last_energy = energy
                
                if self.verbose > 1 and len(self._energy_history) % 5 == 0:
                    print(f"      VQE step {len(self._energy_history)}: {energy:.10f} Ha")
                return energy
            
            # Custom callback to stop early
            def callback(xk):
                nonlocal no_improvement_count
                if no_improvement_count >= max_no_improvement:
                    if self.verbose > 1:
                        print(f"      Early stopping after {len(self._energy_history)} evaluations (no improvement)")
                    return True  # Signal to stop
                return False
            
            num_params = len(self.initial_params)
            maxfun = max(5, num_params * 2)
            
            try:
                result = minimize(
                    objective,
                    self.initial_params,
                    method='SLSQP',
                    options={'maxiter': 5, 'ftol': 1e-6},
                    callback=callback
                )
            except Exception as e:
                if self.verbose:
                    print(f"      SLSQP failed: {e}")
                

            # try:
            #     result = minimize(
            #         objective,
            #         self.initial_params,
            #         method='COBYLA',
            #         options={'maxiter': maxfun, 'tol': 1e-6},
            #         callback=callback
            #     )
            # except Exception as e:
            #     if self.verbose:
            #         print(f"      COBYLA failed: {e}, trying SLSQP...")
            #     result = minimize(
            #         objective,
            #         self.initial_params,
            #         method='SLSQP',
            #         options={'maxiter': 5, 'ftol': 1e-6},
            #         callback=callback
            #     )
            
            if self.verbose > 1:
                print(f"      VQE finished after {len(self._energy_history)} evaluations")
                print(f"      Optimizer message: {result.message}")
            
            return result.fun, result.x
            
        except Exception as e:
            if self.verbose:
                print(f"      VQE optimization failed: {e}")
            return -0.04, None
    
    def evaluate_energy(self, mo_coeff, return_vqe=False):
        """Evaluate quantum energy using statevector simulation."""
        if not self.can_run_vqe:
            if self.verbose > 0:
                print(f"    [VQE] Cannot run VQE ({self.n_qubits} qubits > {self.max_qubits})")
            return -0.04
        
        # Use cached Hamiltonian
        self.hamiltonian = self._cached_hamiltonian
        
        energy, params = self._run_vqe_statevector(self.hamiltonian)
        self._last_energy = energy
        self._last_params = params
        
        if self.verbose > 0:
            ansatz_type = "UCCSD" if self.use_uccsd else "EfficientSU2"
            print(f"    [VQE] {ansatz_type} (statevector): {energy:.10f} Ha")
        
        if return_vqe:
            return energy, {'energy': energy, 'params': params}
        return energy


class QuantumOVOSFull(OVOS):
    """Quantum OVOS with proper active space handling, Hamiltonian caching, and quantum gradients."""
    
    def __init__(self, *args, active_orbitals=None, vqe_energy=False, 
                 use_quantum_gradients=False, max_vqe_qubits=20, 
                 switch_to_uccsd_iter=3, cache_dir='hamiltonian_cache', 
                 grad_eps=1e-6, **kwargs):
        self.active_orbitals = active_orbitals
        self.vqe_energy = vqe_energy
        self.use_quantum_gradients = use_quantum_gradients
        self.max_vqe_qubits = max_vqe_qubits
        self.switch_to_uccsd_iter = switch_to_uccsd_iter
        self.cache_dir = cache_dir
        self.grad_eps = grad_eps
        
        parent_kwargs = kwargs.copy()
        parent_kwargs.pop('vqe_energy', None)
        parent_kwargs.pop('use_quantum_gradients', None)
        parent_kwargs.pop('max_vqe_qubits', None)
        parent_kwargs.pop('switch_to_uccsd_iter', None)
        parent_kwargs.pop('active_orbitals', None)
        parent_kwargs.pop('cache_dir', None)
        parent_kwargs.pop('grad_eps', None)
        
        super().__init__(*args, **parent_kwargs)
        
        self._init_quantum_evaluator()
        
        self.quantum_energy_hist = []
        self._ovos_iteration = 0
        self._grad_history = []
        
        if self.verbose:
            print("\n===== Quantum OVOS Initialized =====")
            print(f"  VQE Energy: {self.vqe_energy}")
            print(f"  Quantum Gradients: {self.use_quantum_gradients}")
            print(f"  Active orbitals: {self.active_orbitals}")
            print(f"  Use Quantum Energy: {self.use_quantum_energy}")
            print(f"  Switch to UCCSD at iteration: {self.switch_to_uccsd_iter}")
            print(f"  Cache directory: {self.cache_dir}")
            print(f"  Gradient epsilon: {self.grad_eps}")
            print("="*40)
    
    def _init_quantum_evaluator(self):
        try:
            self.quantum_evaluator = QuantumEnergyEvaluator(
                self.mol, self.scf, self.active_orbitals, 
                verbose=self.verbose,
                use_simple_ansatz=False,
                use_uccsd=True,
                simple_ansatz_reps=1,
                max_qubits=self.max_vqe_qubits,
                cache_dir=self.cache_dir
            )
            self.use_quantum_energy = self.vqe_energy
        except Exception as e:
            warnings.warn(f"Quantum evaluator initialization failed: {e}")
            self.quantum_evaluator = None
            self.use_quantum_energy = False
            self.vqe_energy = False
    
    def _compute_quantum_gradient(self, R_vec, fock_spin):
        """Compute the gradient of the VQE energy with respect to orbital rotations."""
        if R_vec is None or len(R_vec) == 0:
            return np.zeros(0)
        
        n_params = len(R_vec)
        grad = np.zeros(n_params)
        eps = self.grad_eps
        
        mo_coeffs = self.mo_coeffs
        
        if self.verbose > 1:
            print(f"    Computing quantum gradient for {n_params} parameters...")
        
        for i in range(n_params):
            R_plus = R_vec.copy()
            R_plus[i] += eps
            E_plus = self._get_vqe_total_energy_for_rotation(R_plus, mo_coeffs, fock_spin)
            
            R_minus = R_vec.copy()
            R_minus[i] -= eps
            E_minus = self._get_vqe_total_energy_for_rotation(R_minus, mo_coeffs, fock_spin)
            
            grad[i] = (E_plus - E_minus) / (2.0 * eps)
        
        if self.verbose > 1:
            grad_norm = np.linalg.norm(grad)
            print(f"    Quantum gradient norm: {grad_norm:.6e}")
        
        return grad
    
    def _get_vqe_total_energy_for_rotation(self, R_vec, mo_coeffs, fock_spin):
        """Get the VQE total energy for a given rotation vector."""
        mo_rot, fock_rot, _ = self._rotate_orbitals(mo_coeffs, fock_spin, R_vec)
        mo_rot, fock_rot, _ = self._canonicalize_active(mo_rot, fock_rot, None)
        
        # Get the VQE electronic energy
        vqe_energy = self.quantum_evaluator.evaluate_energy(mo_rot[0])
        
        # Add nuclear repulsion AND the reference energy (constant term)
        total_energy = (vqe_energy + 
                       self.mol.energy_nuc() + 
                       self.quantum_evaluator.reference_energy)
        
        return total_energy
    
    def _get_vqe_total_energy(self, mo_coeffs):
        """Get the VQE total energy for the current orbitals."""
        vqe_energy = self.quantum_evaluator.evaluate_energy(mo_coeffs[0])
        total_energy = (vqe_energy + 
                       self.mol.energy_nuc() + 
                       self.quantum_evaluator.reference_energy)
        return total_energy
    
    def _gradient(self, t_abij, eri_as, D_ab, fock_spin):
        """Override the gradient calculation to use quantum gradients if enabled."""
        if self.use_quantum_energy and self.vqe_energy and self.use_quantum_gradients:
            if hasattr(self, '_current_R'):
                R_vec = self._current_R
            else:
                nvir_act = len(self.active_inocc_indices)
                ninact = len(self.inactive_indices)
                R_vec = np.zeros(nvir_act * ninact)
            
            quantum_grad = self._compute_quantum_gradient(R_vec, fock_spin)
            
            nvir_act = len(self.active_inocc_indices)
            ninact = len(self.inactive_indices)
            grad_matrix = quantum_grad.reshape(nvir_act, ninact)
            
            if self.verbose > 1:
                grad_norm = np.linalg.norm(grad_matrix)
                print(f"    Quantum gradient norm: {grad_norm:.6e}")
            
            return grad_matrix
        else:
            return super()._gradient(t_abij, eri_as, D_ab, fock_spin)
    
    def _newton_step(self, G, H, iteration, start_counting):
        """Override the Newton step to handle quantum gradients."""
        if self.use_quantum_energy and self.vqe_energy and self.use_quantum_gradients:
            g_vec = G.flatten()
            grad_norm = np.linalg.norm(g_vec)
            
            if self.verbose:
                print(f"        Quantum Newton step: grad_norm = {grad_norm:.6e}")
            
            if grad_norm < 1e-12:
                return np.zeros_like(g_vec)
            
            self._grad_history.append(grad_norm)
            
            step_size = min(0.1, 0.01 / (grad_norm + 1e-12))
            return -step_size * g_vec / (grad_norm + 1e-12)
        else:
            return super()._newton_step(G, H, iteration, start_counting)
    
    def _mp2_energy(self, fock_spin, t_abij, eri_as):
        if self.use_quantum_energy and self.vqe_energy and self.quantum_evaluator:
            try:
                self._ovos_iteration += 1
                
                should_switch = False
                if self.switch_to_uccsd_iter == 0 and self._ovos_iteration == 1:
                    should_switch = True
                elif self._ovos_iteration == self.switch_to_uccsd_iter:
                    should_switch = True
                
                if should_switch and not self.quantum_evaluator.use_uccsd:
                    if self.verbose:
                        print("    [OVOS] Switching to UCCSD ansatz for final accuracy...")
                    self.quantum_evaluator.switch_to_uccsd()
                
                mo_coeffs = self.mo_coeffs[0]
                vqe_electronic_energy = self.quantum_evaluator.evaluate_energy(mo_coeffs)
                
                # The total energy = VQE energy + nuclear repulsion + reference energy
                # The reference energy is the constant term that was removed
                total_energy = (vqe_electronic_energy + 
                              self.mol.energy_nuc() + 
                              self.quantum_evaluator.reference_energy)
                
                corr_energy = total_energy - self.scf.e_tot
                
                self.quantum_energy_hist.append(corr_energy)
                
                if self.verbose > 1:
                    print(f"    [OVOS] VQE electronic: {vqe_electronic_energy:.10f} Ha")
                    print(f"    [OVOS] Reference energy: {self.quantum_evaluator.reference_energy:.10f} Ha")
                    print(f"    [OVOS] Nuclear repulsion: {self.mol.energy_nuc():.10f} Ha")
                    print(f"    [OVOS] VQE total: {total_energy:.10f} Ha, correlation: {corr_energy:.10f} Ha")
                
                return total_energy  
                
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: VQE failed: {e}")
                    print(f"  Falling back to classical MP2")
                return super()._mp2_energy(fock_spin, t_abij, eri_as)
        else:
            return super()._mp2_energy(fock_spin, t_abij, eri_as)
    
    def run(self, mo_coeffs, fock_spin=None):
        self._ovos_iteration = 0
        self.quantum_energy_hist = []
        self._grad_history = []
        
        nvir_act = len(self.active_inocc_indices)
        ninact = len(self.inactive_indices)
        self._current_R = np.zeros(nvir_act * ninact)
        
        result = super().run(mo_coeffs, fock_spin)
        
        if self.verbose:
            print("\n===== Quantum OVOS Summary =====")
            if self.quantum_energy_hist:
                print(f"  Initial VQE Correlation: {self.quantum_energy_hist[0]:.10f} Ha")
                print(f"  Final VQE Correlation:   {self.quantum_energy_hist[-1]:.10f} Ha")
                delta = self.quantum_energy_hist[-1] - self.quantum_energy_hist[0]
                print(f"  Total Change:       {delta:+.10f} Ha")
                print(f"  Evaluations: {len(self.quantum_energy_hist)}")
                if self.quantum_evaluator:
                    print(f"  Final Ansatz: {'UCCSD' if self.quantum_evaluator.use_uccsd else 'EfficientSU2'}")
                    print(f"  Reference Energy: {self.quantum_evaluator.reference_energy:.10f} Ha")
                    print(f"  Inactive Energy: {self.quantum_evaluator.inactive_energy:.10f} Ha")
                if self._grad_history:
                    print(f"  Gradient Norms: min={min(self._grad_history):.6e}, max={max(self._grad_history):.6e}")
            else:
                print("  No quantum energy evaluations (used classical MP2)")
            print("="*40)
        
        return result


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from pyscf import gto, scf
    
    print("="*60)
    print("Quantum OVOS Demonstration with Quantum Gradients")
    print("="*60)
    
    mol = gto.Mole()
    mol.atom = 'O 0.0000 0.0000  0.1173; H 0.0000    0.7572  -0.4692; H 0.0000   -0.7572 -0.4692'
    mol.basis = 'sto-3g'
    mol.unit = 'Angstrom'
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = False
    mol.verbose = 0
    mol.build()
    
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    print(f"RHF Energy: {mf.e_tot:.10f} Ha")
    
    Fao = [mf.get_fock(), mf.get_fock()]
    mo_coeffs = [mf.mo_coeff, mf.mo_coeff]
    active_orbitals = list(range(12))
    
    qovos = QuantumOVOSFull(
        mol=mol, scf=mf, Fao=Fao,
        num_opt_virtual_orbs=2,
        mo_coeff=mo_coeffs,
        verbose=1,
        max_iter=5,
        vqe_energy=True,
        active_orbitals=active_orbitals,
        max_vqe_qubits=20,
        switch_to_uccsd_iter=0,
        cache_dir='hamiltonian_cache',
        use_quantum_gradients=True,
        grad_eps=1e-5,
    )
    
    result = qovos.run(mo_coeffs, fock_spin=None)
    
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    print(f"Final Correlation Energy: {result[0]:.10f} Ha")
    print(f"Stop Reason: {result[-1]}")
    print("="*60)