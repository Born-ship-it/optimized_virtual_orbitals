

TOC copy-paste:

xii 
9.8.4 The error function 
9.8.5 The complementary error function 
9.8.6 The confluent hypergeometric function 
9.9 The McMurchie-Davidson scheme for Coulomb integrals 
9.9.1 Hermite Coulomb integrals 
9.9.2 The evaluation of Hermite Coulomb integrals 
9.9.3 Cartesian Coulomb integrals by Hermite expansion 
9.9.4 Cartesian Coulomb integrals by Hermite recursion 
9.9.5 Computational considerations for the one-electron integrals 
9.9.6 Computational considerations for the two-electron integrals 
9.10  The Obara-Saika scheme for Coulomb integrals 
9.10.1  The Obara-Saika scheme for one-electron Coulomb integrals 
9.10.2  The Obara-Saika scheme for two-electron Coulomb integrals 
9.10.3  The electron-transfer and horizontal recurrence relations 
9.10.4  Computational considerations for the two-electron integrals 
9.11  Rys  quadrature for Coulomb integrals 
9.11.1  Motivation for the Gaussian-quadrature scheme 
9.11.2  Gaussian quadrature for even polynomials and weight functions 
9.11.3  Rys  polynomials and Gauss-Rys quadrature 
9.11.4  The Rys  scheme for Hermite Coulomb integrals 
9.11.5  The Rys  scheme for Cartesian Coulomb integrals 
9.11.6  Obara-Saika recursion for the two-dimensional Rys  integrals 
9.11.7  Computational considerations for the two-electron integrals 
9.12  Scaling properties of the molecular integrals 
9.12.1  Linear scaling of the overlap and kinetic-energy integrals 
9.12.2  Quadratic scaling of the Coulomb integrals 
9.12.3  Linear scaling of the nonclassical Coulomb integrals 
9.12.4  The Schwarz inequality 
9.13  The multipole method for Coulomb integrals 
9.13.1  The multipole method for primitive two-electron integrals 
9.13.2  Convergence of the multipole expansion 
9.13.3  The multipole method for contracted two-electron integrals 
9.13.4  Translation of multipole moments 
9.13.5  Real multipole moments 
9.13.6  The real translation matrix 
9.13.7  The real interaction matrix 
9.13.8  Evaluation of the scaled solid harmonics 
9.14  The multipole method for large systems 
9.14.1  The naive multipole method 
9.14.2  The two-level multipole method 
9.14.3  The fast multipole method 
9.14.4  The continuous fast multipole method 
References 
Further reading 
Exercises 
Solutions 
CONTENTS 
369 370 371 
372 
373 
374 
375 
377 
377 
379 
381 
382 
383 
385 
386 
387 
388 
388 390 392 
394 

395 
397 
398 
398 400 401 403 405 405 409 409 410 
412 
413 
414 
415 
417 
417 420 421 
423 
425 
426 
426 
428  
CONTENTS xiii 
10  Hartree-Fock Theory 433 
10.1  Parametrization of the wave function and the energy 433 
10.1.1 Singlet and triplet CSFs 434 
10.1.2 Orbital rotations 435 
10.1.3 Expansion of the energy 437 
10.2  The Hartree-Fock wave function 438 
10.2.1 The Hartree-Fock wave function 438 
10.2.2 Redundant parameters 440 10.2.3 The Brillouin theorem 441 
10.2.4 Size-extensivity 442 
10.3  Canonical Hartree-Fock theory 443 
10.3.1 The Fock operator 444 
10.3.2 Identification of the elements of the Fock operator 445 
10.3.3 The Fock matrix 447 
10.3.4 The self-consistent field method 448 
10.3.5 The variational and canonical conditions compared 449 
10.4  The RHF total energy and orbital energies 450 10.4.1 The Hamiltonian and the Fock operator 450 10.4.2 The canonical representation and orbital energies 450 10.4.3 The Hartree-Fock energy 452 
10.4.4 Hund's rule for singlet and triplet states 452 
10.4.5 The fluctuation potential 453 
10.5  Koopmans'  theorem 454 
10.5.1 Koopmans'  theorem for ionization potentials 454 
10.5.2 Koopmans'  theorem for electron affinities 455 
10.5.3 Ionization potentials of H20  and N2 456 
10.6  The Roothaan - Hall self-consistent field equations 458 
10.6.1 The Roothaan-Hall equations 458 
10.6.2 DIIS convergence acceleration 460 10.6.3 Integral-direct Hartree-Fock theory 463 
10.7  Density-based Hartree-Fock theory 465 
10.7.1 Density-matrix formulation of Hartree-Fock theory 465 
10.7.2 Properties of the MO density matrix 466 
10.7.3 Properties of the AO density matrix 467 
10.7.4 Exponential parametrization of the AO density matrix 468 
10.7.5 The redundancy of the exponential parametrization 469 
10.7.6 Purification of the density matrix 470 10.7.7 Convergence of the purification scheme 471 
10.7.8 The Hartree-Fock energy and the variational conditions 473 
10.7.9 The density-based SCF method 475 
10.7.10  Optimization of the SCF orbital-energy function 477 
10.7.11  Linear scaling of the density-based SCF scheme 477 
10.8  Second-order optimization 478 
10.8.1 Newton's method 478 
10.8.2 Density-based formulation of Newton's method 480 10.8.3 The electronic gradient in orbital-based Hartree-Fock theory 481  
xiv 
10.8.4 The inactive and active Fock matrices 
10.8.5 Computational cost for the calculation of the Fock matrix 
10.8.6 The electronic Hessian in orbital-based Hartree-Fock theory 
10.8.7 Linear transformations in the MO basis 
10.8.8 Linear transformations in the AO basis 
10.9 The SCF method as  an approximate second-order method 
10.9.1 The GBT vector 
10.9.2 The Fock operator 
10.9.3 Identification from the gradient 
10.9.4 Identification from the Hessian 
10.9.5 Convergence rates 
10.9.6 The SCF and Newton methods compared 
10.10  Singlet and triplet instabilities in RHF theory 
10.10.1  Orbital-rotation operators in RHF and UHF theories 
10.10.2  RHF instabilities for nondegenerate electronic states 
10.10.3  RHF energies of degenerate electronic states 
10.10.4  Triplet instabilities in H2 
10.10.5  Triplet instabilities in H20 10.10.6  Singlet instabilities in the allyl radical 
10.11  Multiple solutions in Hartree-Fock theory 
References 
Further reading 
Exercises 
Solutions 
11  Configuration-Interaction Theory 
11.1 The CI model 
11.1.1 The CI model 
11.1.2 Full CI wave functions 
11.1.3 Truncated CI wave functions:  CAS  and RAS expansions 
11.2 Size-extensivity and the CI model 
11.2.1 FCI wave functions 
11.2.2 Truncated CI wave functions 
11.2.3 The Davidson correction 
11.2.4 A numerical study of size-extensivity 
11.3 A CI model system for noninteracting hydrogen molecules 
11.3.1 The CID wave function and energy 
11.3.2 The Davidson correction 
11.3.3 The CID one-electron density matrix 
11.3.4 The FCI distribution of excitation levels 
11.4 Parametrization of the CI model 
11.4.1 The CI expansion 
11.4.2 The CI energy 
11.5 Optimization of the CI wave function 
11.5.1 The Newton step 
11.5.2 Convergence rate of Newton's method for the CI energy 
CONTENTS 
482 
484 
485 
488 
489 490 491 
491 
493 
494 
494 
496 
496 
497 
498 
499 500 500 502 504 506 506 506 513 
523 
523 
524 
524 
526 
527 
528 
529 530 531 
535 
535 
537 
537 
538 540 540 542 
543 
544 
545 
CONTENTS 11.5.3  Approximate Newton schemes 
11.5.4  Convergence rate of quasi-Newton schemes for the CI energy 
11.6 Slater determinants as  products of alpha and beta strings 
11.7 The determinantal representation of the Hamiltonian operator 
11.8 Direct CI methods 
11.8.1  General considerations 
11.8.2  Ordering and addressing of spin strings 
11.8.3  The N-resolution method 
11.8.4  The minimal operation-count method 
11.8.5  Direct CI algorithms for RAS  calculations 
11.8.6  Simplifications for wave functions  of zero projected spin 
11.8.7  Density matrices 
11.9 CI orbital transformations 
11.10  Symmetry-broken CI solutions 
References 
Further reading 
Exercises 
Solutions 
12  Multiconfigurational Self-Consistent Field Theory 
12.1 The MCSCF model 
12.2 The MCSCF energy and wave function 
12.2.1  The parametrization of the MCSCF state 
12.2.2  The Taylor expansion of the MCSCF energy 
12.2.3  The MCSCF electronic gradient and Hessian 
12.2.4  Invariance of the second-order MCSCF energy 
12.2.5  Rank-l contributions to  the MCSCF electronic Hessian 
12.2.6  Redundant orbital rotations 
12.2.7  The MCSCF electronic gradient at stationary points 
12.2.8  The MCSCF electronic Hessian at stationary points 
12.3 The MCSCF Newton trust-region method 
12.3.1  The Newton step 
12.3.2  The level-shifted Newton step 
12.3.3  The level-shift parameter 
12.3.4  Step control for ground states 
12.3.5  Step control for excited states 
12.3.6  Trust-radius update schemes 
12.4 The Newton eigenvector method 
12.4.1  The MCSCF eigenvalue problem 
12.4.2  The Newton eigenvector method 
12.4.3  Norm-extended optimization 
12.4.4  The augmented-Hessian method 
12.5 Computational considerations 
12.5.1  The MCSCF electronic gradient 
12.5.2  MCSCF Hessian transformations 
12.5.3  Inner and outer iterations 
xv 
547 
548 550 552 
554 
554 
555 
558 560 
564 
567 
568 
569 
573 
574 
575 
575 
583 
598 
598 600 600 601 603 604 604 605 608 609 610 
610 
611 
612 
614 
614 
615 
616 
616 
617 
619 620 621 
622 
623 
625  
xvi 
12.5.4  The structure of the MCSCF electronic Hessian 
12.5.5  Examples of MCSCF optimizations 
12.6  Exponential parametrization of the configuration space 
12.6.1  General exponential parametrization of the configuration space 
12.6.2  Exponential parametrization for a single reference state 
12.6.3  A basis for the orthogonal complement to  the reference state 
12.6.4  Exponential parametrization for several reference states 
12.7  MCSCF theory for several electronic states 
12.7.1  Separate optimization of the individual states 
12.7.2  State-averaged MCSCF theory 
12.8  Removal of RHF instabilities in MCSCF theory 
12.8.1  Bond breaking in H20 12.8.2  The ground state of the allyl radical 
References 
Further reading 
Exercises 
Solutions 
13  Coupled-Cluster Theory 
13.1  The coupled-cluster model 
13.1.1  Pair clusters 
13 .1.2  The coupled-cluster wave function 
13 .1.3  Connected and disconnected clusters 
13 .1.4  The coupled-cluster Schrodinger equation 
13.2  The coupled-cluster exponential ansatz 
13 .2.1  The exponential ansatz 
13.2.2  The coupled-cluster hierarchy of excitation levels 
13.2.3  The projected coupled-cluster equations 
13.2.4  The coupled-cluster energy 
13.2.5  The coupled-cluster amplitude equations 
13.2.6  Coupled-cluster theory in the canonical representation 
13.2.7  Comparison of the CI and coupled-cluster hierarchies 
13.2.8  Cluster-commutation conditions and operator ranks 
13.3  Size-extensivity in coupled-cluster theory 
13.3.1  Size-extensivity in linked coupled-cluster theory 
13.3.2  Termwise size-extensivity 
13.3.3  Size-extensivity in unlinked coupled-cluster theory 
13.3.4  A numerical study of size-extensivity 
13.4  Coupled-cluster optimization techniques 
13.4.1  Newton's method 
13.4.2  The perturbation-based quasi-Newton method 
13.4.3  DIIS  acceleration of the quasi-Newton method 
13.4.4  Examples of coupled-cluster optimizations 
CONTENTS 
626 
628 630 630 631 
633 
634 
637 
637 
638 640 640 641 
643 
643 
643 
645 
648 
648 
649 650 650 651 
654 
654 
654 
657 660 660 662 
662 
663 
665 
665 
667 
668 
669 670 671 
672 
672 
673  
CONTENTS 
13.5  The coupled-cluster variational Lagrangian 
13.5.1  The coupled-cluster Lagrangian 
13.5.2  The Hellmann-Feynman theorem 
13.5.3  Lagrangian density matrices 
13.6  The equation-of-motion coupled-cluster method 
13.6.1  The equation-of-motion coupled-cluster model 
13.6.2  The EOM-CC eigenvalue problem 
13.6.3  The similarity-transformed Hamiltonian and the Jacobian 
13.6.4  Solution of the EOM-CC eigenvalue problem 
13.6.5  Size-extensivity of the EOM-CC energies 
13.6.6  Final comments 
13.7  The closed-shell CCSD model 
13.7.1  Parametrization of the CCSD cluster operator 
13.7.2  The CCSD energy expression 
13.7.3  The Tl-transformed Hamiltonian 
13.7.4  The Tl-transformed integrals 
13.7.5  Representation of the CCSD projection manifold 
13.7.6  The norm of the CCSD wave function 
13.7.7  The CCSD singles projection 
13.7.8  The CCSD doubles projection 
13.7.9  Computational considerations 
13.8  Special treatments of coupled-cluster theory 
13.8.1  Orbital-optimized and Brueckner coupled-cluster theories 
13.8.2  Quadratic configuration-interaction theory 
13.9  High-spin open-shell coupled-cluster theory 
13.9.1  Spin-restricted coupled-cluster theory 
13.9.2  Total spin of the spin-restricted coupled-cluster wave function 
13.9.3  The projection manifold in spin-restricted theory 
13.9.4  Spin-adapted CCSD theory 
References 
Further reading 
Exercises 
Solutions 
14  Perturbation Theory 
14.1  Rayleigh-SchrMinger perturbation theory 
14.1.1  RSPT energies and wave functions 
14.1.2  Wigner's 2n + 1 rule 14.1.3  The Hylleraas functional 14.1.4  Size-extensivity in RSPT 14.2  M\Zlller-Plesset perturbation theory 
14.2.1  The zero-order MPPT system 
14.2.2  The MPI wave function 
xvii 
674 
674 
675 
676 
677 
677 
679 680 681 
683 
684 
685 
685 
686 
687 690 691 
692 
693 
695 
697 
698 
698 702 704 704 707 708 709 711 
712 
712 
717 
724 
725 
726 
728 
734 
736 
739 740 741  
xviii 
14.2.3  The MP2 wave function 
14.2.4  The M\Zlller-Plesset energies 
14.2.5  Explicit expressions for MPPT wave functions  and energies 
14.2.6  Size-extensivity in M\Zlller-Plesset theory 
14.3  Coupled-cluster perturbation theory 
14.3.1  The similarity-transformed exponential ansatz of CCPT 14.3.2  The CCPT amplitude equations 
14.3.3  The CCPT wave functions 
14.3.4  The CCPT energies 
14.3.5  Size-extensivity in CCPT 14.3.6  The CCPT Lagrangian 
14.3.7  The CCPT variational equations 
14.3.8  CCPT energies that obey the 2n + 1 rule 14.3.9  Size-extensivity of the CCPT Lagrangian 
14.4  M\Zlller-Plesset theory for closed-shell systems 
14.4.1  The closed-shell zero-order system 
14.4.2  The closed-shell variational Lagrangian 
14.4.3  The closed-shell wave-function corrections 
14.4.4  The closed-shell energy corrections 
14.5  Convergence in perturbation theory 
14.5.1  A two-state model 
14.5.2  Conditions for convergence 
14.5.3  Intruders in the general two-state model 
14.5.4  Prototypical intruders 
14.5.5  Convergence of the M\Zlller-Plesset series 
14.5.6  Analytic continuation 
14.6  Perturbative treatments of coupled-cluster wave functions 
14.6.1  Perturbation analysis of the coupled-cluster hierarchy 
14.6.2  Iterative hybrid methods 
14.6.3  Noniterative hybrid methods:  the CCSD(T) model 
14.6.4  Hybrid and nonhybrid methods compared 
14.7  Multiconfigurational perturbation theory 
14.7.1  The zero-order CASPT Hamiltonian 
14.7.2  Size-extensivity in CASPT 14.7.3  The CASPT wave function and energy 
14.7.4  Sample CASPT calculations 
References 
Further reading 
Exercises 
Solutions 
15  Calibration of the  Electronic-Structure Models 
15.1  The sample molecules 
15.2  Errors in quantum-chemical calculations 
15.2.1  Apparent and intrinsic errors 
15.2.2  Statistical measures of errors 
CONTENTS 
742 
745 
746 
747 
749 
749 
751 
752 
753 
754 
755 
756 
758 
759 
759 760 761 
763 
766 
769 770 771 
772 
776 
778 
782 
783 
784 
789 
793 
795 
796 
796 
798 800 801 803 804 804 809 
817 
817 
819 
819 
821 
CONTENTS xix 
IS.3 Molecular equilibrium structures: bond distances 821 
IS.3.1  Experimental bond distances 822 
IS.3.2  Mean errors and standard deviations 822 
IS.3.3  Normal distributions 82S 
IS.3.4  Mean absolute deviations 82S 
IS.3.S  Maximum errors 827 
IS.3.6  The CCSDT and CCSD(T) models 827 
IS.3.7  The effect of core correlation on bond distances 828 
IS.3.8  Trends in the convergence towards experiment 829 
IS.3.9  Summary 831 
IS.4 Molecular equilibrium structures: bond angles 832 
IS.4.1  Experimental bond angles 832 
IS.4.2  Calculated bond angles 833 
IS.4.3  Summary 834 
IS.S Molecular dipole moments 836 
IS.S.1  Experimental dipole moments 836 
IS.S.2  Calculated dipole moments 837 
IS.S.3  Predicted dipole moments 838 
IS.S.4  Analysis of the calculated dipole moments 839 
IS.S.S  Summary 840 
IS.6 Molecular and atomic energies 840 
IS.6.1  The total electronic energy 841 
IS.6.2  Contributions to  the total electronic energy 842 
IS.6.3  Basis-set convergence 844 
IS.6.4  CCSDT corrections 846 
IS.6.S  Molecular vibrational corrections 847 
IS.6.6  Relativistic corrections 849 
IS.6.7  Summary 8S3 
IS.7 Atomization energies 8S4 
IS.7.1  Experimental atomization energies 8S4 
IS.7.2  Statistical analysis of atomization energies 8SS IS.7.3  Extrapolation of atomization energies 8S8 
IS.7.4  Core contributions to  atomization energies 861 
IS.7.S  CCSDT corrections 863 
IS.7.6  Summary 864 
IS.8 Reaction enthalpies 86S 
IS.8.1  Experimental reaction enthalpies 86S 
IS.8.2  Statistical analysis of reaction enthalpies 867 
IS.8.3  Extrapolation and convergence to  the basis-set limit 870 
IS.8.4  Core contributions to reaction enthalpies 871 
IS.8.S  Summary 873 
IS.9 Conformational barriers 874 
IS.9.1  The barrier to  linearity of water 87S 
IS.9.2  The inversion barrier of ammonia 876 
IS.9.3  The torsional barrier of ethane 877 
IS.9.4  Summary 879 
IS.1O  Conclusions 879  
xx CONTENTS References 882 
Further reading 882 
Exercises 882 
Solutions 883 
List of Acronyms 885 
Index 887