# AI Coding Agent Instructions - OVOS Project

Always show code corrections in chat only. Do not attempt to edit files directly.

## Academic Guidelines
Using generative AI for "limited extent" per UCPH guidelines: idea generation, structure suggestions, literature search, clarifications. Follow good scientific practice - document AI usage, verify all outputs, understand the physics and chemistry.

---

## Project Context
Master's thesis on **Optimized Virtual Orbitals (OVO)** for quantum computing at UCPH. Implementation based on [L. Adamowicz & R. J. Bartlett (1987)](https://pubs.aip.org/aip/jcp/article/86/11/6314/93345/Optimized-virtual-orbital-space-for-high-level) - minimizes second-order correlation energy (MP2) using orbital rotations.

### Reference
**Optimized virtual orbital space for high-level correlated calculations**  
Adamowicz, L. & Bartlett, R. J.  
*J. Chem. Phys.* **86**, 6314-6324 (1987)  
DOI: [10.1063/1.452468](https://doi.org/10.1063/1.452468)

### Core Concept

Reduce the virtual orbital space dimension from N_VIRT to N'_VIRT < N_VIRT while preserving most correlation energy. Uses second-order Hylleraas functional to optimize orbital rotations between active and inactive virtual spaces.

**Key Result**: Can recover ~90% of correlation energy with 50% of virtual orbitals, or ~100% when combining OVOS result with exact E₂.

#### Method Summary
Optimize a reduced virtual orbital space by rotating active virtuals against inactive virtuals to minimize second-order correlation energy via the Hylleraas functional.

**MP1 Amplitudes**:
The article defines MP1 amplitudes as:
\[t_{ij}^{ab} = - \frac{<ab|ij>}{- \epsilon_i - \epsilon_j + \epsilon_a + \epsilon_b}\],
where \(i,j\) are occupied orbitals, \(a,b\) are virtual orbitals, and \(\epsilon\) are orbital energies.
Note: Am not sure if <ab|ij> means chemist's or physicist's notation, or if it is the antisymmetrized integral.

**MP2 Correlation Energy**:
The article gives the MP2 correlation energy as:
\[J_2 = \sum_{i>j} J_{ij}^{(2)}\],
where
\[J_{ij}^{(2)} = \sum_{a>b} \sum_{c>d} t_{ij}^{ab} t_{ij}^{cd} [(f_{ac} \delta_{bd} - f_{ad} \delta_{bc}) + (f_{bd} \delta_{ac} - f_{bc} \delta_{ad})] - (\epsilon_i + \epsilon_j)(\delta_{ac}\delta_{bd} - \delta_{ad}\delta_{bc}) + 2 \sum_{a>b} t_{ij}^{ab} <ab|ij>  \],
where \(f_{ac}\) are Fock matrix elements in the MO basis, and \(\delta\) is the Kronecker delta, and \(<ab|ij>\) are two-electron integrals, and \(i,j\) are occupied orbitals, \(a,b,c,d\) are virtual orbitals, and \(\epsilon\) are orbital energies.
Note: Am not sure if <ab|ij> means chemist's or physicist's notation, or if it is the antisymmetrized integral.

**Gradient of J₂ wrt orbital rotations**:
The gradient of J₂ with respect to orbital rotations between active virtual orbitals \(a\) and inactive virtual orbitals \(r\) is given by:


**PySCF Package, highligtning some usefull function:**
PySCF comes with functions to convert between spatial and spin orbital representations:

```python
from pyscf.cc.addons import spatial2spin, spin2spatial
# Convert spatial MO coefficients to spin MO coefficients
mo_coeff_spin = spatial2spin(mol, mo_coeff_spatial)
# Convert spin MO coefficients back to spatial MO coefficients
mo_coeff_spatial = spin2spatial(mol, mo_coeff_spin)
```
This is useful when working with methods that require spin orbitals, such as CCSD or MP2, while the initial calculations may be performed in spatial orbital basis.

I detail here the code used in spatial2spin and spin2spatial for reference:

```python
import numpy
from pyscf import lib
from pyscf.cc.bccd import bccd_kernel_


[docs]
def spatial2spin(tx, orbspin=None):
    '''Convert T1/T2 of spatial orbital representation to T1/T2 of
    spin-orbital representation
    '''
    if isinstance(tx, numpy.ndarray) and tx.ndim == 2:
        # RCCSD t1 amplitudes
        return spatial2spin((tx,tx), orbspin)
    elif isinstance(tx, numpy.ndarray) and tx.ndim == 4:
        # RCCSD t2 amplitudes
        t2aa = tx - tx.transpose(1,0,2,3)
        return spatial2spin((t2aa,tx,t2aa), orbspin)
    elif len(tx) == 2:  # t1
        t1a, t1b = tx
        nocc_a, nvir_a = t1a.shape
        nocc_b, nvir_b = t1b.shape
    elif len(tx) == 3:  # t2
        t2aa, t2ab, t2bb = tx
        nocc_a, nocc_b, nvir_a, nvir_b = t2ab.shape
    else:
        raise RuntimeError('Unknown T amplitudes')

    if orbspin is None:
        assert nocc_a == nocc_b
        orbspin = numpy.zeros((nocc_a+nvir_a)*2, dtype=int)
        orbspin[1::2] = 1

    nocc = nocc_a + nocc_b
    nvir = nvir_a + nvir_b
    idxoa = numpy.where(orbspin[:nocc] == 0)[0]
    idxob = numpy.where(orbspin[:nocc] == 1)[0]
    idxva = numpy.where(orbspin[nocc:] == 0)[0]
    idxvb = numpy.where(orbspin[nocc:] == 1)[0]

    if len(tx) == 2:  # t1
        t1 = numpy.zeros((nocc,nvir), dtype=t1a.dtype)
        lib.takebak_2d(t1, t1a, idxoa, idxva)
        lib.takebak_2d(t1, t1b, idxob, idxvb)
        t1 = lib.tag_array(t1, orbspin=orbspin)
        return t1

    else:
        t2 = numpy.zeros((nocc**2,nvir**2), dtype=t2aa.dtype)
        idxoaa = idxoa[:,None] * nocc + idxoa
        idxoab = idxoa[:,None] * nocc + idxob
        idxoba = idxob[:,None] * nocc + idxoa
        idxobb = idxob[:,None] * nocc + idxob
        idxvaa = idxva[:,None] * nvir + idxva
        idxvab = idxva[:,None] * nvir + idxvb
        idxvba = idxvb[:,None] * nvir + idxva
        idxvbb = idxvb[:,None] * nvir + idxvb
        t2aa = t2aa.reshape(nocc_a*nocc_a,nvir_a*nvir_a)
        t2ab = t2ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
        t2bb = t2bb.reshape(nocc_b*nocc_b,nvir_b*nvir_b)
        lib.takebak_2d(t2, t2aa, idxoaa.ravel()  , idxvaa.ravel()  )
        lib.takebak_2d(t2, t2bb, idxobb.ravel()  , idxvbb.ravel()  )
        lib.takebak_2d(t2, t2ab, idxoab.ravel()  , idxvab.ravel()  )
        lib.takebak_2d(t2, t2ab, idxoba.T.ravel(), idxvba.T.ravel())
        abba = -t2ab
        lib.takebak_2d(t2, abba, idxoab.ravel()  , idxvba.T.ravel())
        lib.takebak_2d(t2, abba, idxoba.T.ravel(), idxvab.ravel()  )
        t2 = lib.tag_array(t2, orbspin=orbspin)
        return t2.reshape(nocc,nocc,nvir,nvir)



spatial2spinorb = spatial2spin


[docs]
def spin2spatial(tx, orbspin):
    '''Convert T1/T2 in spin-orbital basis to T1/T2 in spatial orbital basis
    '''
    if tx.ndim == 2:  # t1
        nocc, nvir = tx.shape
    elif tx.ndim == 4:
        nocc, nvir = tx.shape[1:3]
    else:
        raise RuntimeError('Unknown T amplitudes')

    idxoa = numpy.where(orbspin[:nocc] == 0)[0]
    idxob = numpy.where(orbspin[:nocc] == 1)[0]
    idxva = numpy.where(orbspin[nocc:] == 0)[0]
    idxvb = numpy.where(orbspin[nocc:] == 1)[0]
    nocc_a = len(idxoa)
    nocc_b = len(idxob)
    nvir_a = len(idxva)
    nvir_b = len(idxvb)

    if tx.ndim == 2:  # t1
        t1a = lib.take_2d(tx, idxoa, idxva)
        t1b = lib.take_2d(tx, idxob, idxvb)
        return t1a, t1b
    else:
        idxoaa = idxoa[:,None] * nocc + idxoa
        idxoab = idxoa[:,None] * nocc + idxob
        idxobb = idxob[:,None] * nocc + idxob
        idxvaa = idxva[:,None] * nvir + idxva
        idxvab = idxva[:,None] * nvir + idxvb
        idxvbb = idxvb[:,None] * nvir + idxvb
        t2 = tx.reshape(nocc**2,nvir**2)
        t2aa = lib.take_2d(t2, idxoaa.ravel(), idxvaa.ravel())
        t2bb = lib.take_2d(t2, idxobb.ravel(), idxvbb.ravel())
        t2ab = lib.take_2d(t2, idxoab.ravel(), idxvab.ravel())
        t2aa = t2aa.reshape(nocc_a,nocc_a,nvir_a,nvir_a)
        t2bb = t2bb.reshape(nocc_b,nocc_b,nvir_b,nvir_b)
        t2ab = t2ab.reshape(nocc_a,nocc_b,nvir_a,nvir_b)
        return t2aa,t2ab,t2bb

```


PySCf also comes with, general Integral transformation module:

```python
import tempfile
import numpy
import h5py
from pyscf import gto
from pyscf.ao2mo import incore
from pyscf.ao2mo import outcore
from pyscf.ao2mo import r_outcore
from pyscf.ao2mo.addons import load, restore


def full(eri_or_mol, mo_coeff, erifile=None, dataname='eri_mo', intor='int2e',
         *args, **kwargs):
    r'''MO integral transformation. The four indices (ij|kl) are transformed
    with the same set of orbitals.

    Args:
        eri_or_mol : ndarray or Mole object
            If AO integrals are given as ndarray, it can be either 8-fold or
            4-fold symmetry.  The integral transformation are computed incore
            (ie all intermediate are held in memory).
            If Mole object is given, AO integrals are generated on the fly and
            outcore algorithm is used (ie intermediate data are held on disk).
        mo_coeff : ndarray
            Orbital coefficients in 2D array
        erifile : str or h5py File or h5py Group object
            *Note* this argument is effective when eri_or_mol is Mole object.
            The file to store the transformed integrals.  If not given, the
            transformed integrals are held in memory.

    Kwargs:
        dataname : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        intor : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            *Note* this argument is effective when eri_or_mol is Mole object.
            Permutation symmetry for the AO integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry
            | 'a4ij' : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a4kl' : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | 'a2ij' : anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a2kl' : anti-symmetry between k, l in (ij|kl) (TODO)

        comp : int
            *Note* this argument is effective when eri_or_mol is Mole object.
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        max_memory : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The maximum size of cache to use (in MB), large cache may **not**
            improve performance.
        ioblk_size : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The block size for IO, large block size may **not** improve performance
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four orbital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals

    Returns:
        If eri_or_mol is array or erifile is not give,  the function returns 2D
        array (or 3D array if comp > 1) of transformed MO integrals.  The MO
        integrals may or may not have the permutation symmetry (controlled by
        the kwargs compact).
        Otherwise, return the file/fileobject where the MO integrals are saved.


    Examples:

    >>> from pyscf import gto, ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     with h5py.File(h5file, 'r') as f5:
    ...         print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))

    >>> eri1 = ao2mo.full(mol, mo1)
    >>> print(eri1.shape)
    (55, 55)

    >>> eri = mol.intor('int2e_sph', aosym='s8')
    >>> eri1 = ao2mo.full(eri, mo1, compact=False)
    >>> print(eri1.shape)
    (100, 100)

    >>> ao2mo.full(mol, mo1, 'full.h5')
    >>> view('full.h5')
    dataset ['eri_mo'], shape (55, 55)

    >>> ao2mo.full(mol, mo1, 'full.h5', dataname='new', compact=False)
    >>> view('full.h5', 'new')
    dataset ['eri_mo', 'new'], shape (100, 100)

    >>> ao2mo.full(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)

    >>> ao2mo.full(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    if isinstance(eri_or_mol, numpy.ndarray):
        return incore.full(eri_or_mol, mo_coeff, *args, **kwargs)
    elif isinstance(eri_or_mol, gto.MoleBase):
        if '_spinor' in intor:
            mod = r_outcore
        else:
            mod = outcore

        if isinstance(erifile, (str, h5py.Group)): # args[0] is erifile
            return mod.full(eri_or_mol, mo_coeff, erifile, dataname, intor,
                            *args, **kwargs)
        elif isinstance(erifile, tempfile._TemporaryFileWrapper):
            return mod.full(eri_or_mol, mo_coeff, erifile.name, dataname, intor,
                            *args, **kwargs)
        else:
            return mod.full_iofree(eri_or_mol, mo_coeff, intor, *args, **kwargs)
    else:
        raise RuntimeError('ERI is not available. If this is generated by mf._eri, '
                           'the integral tensor is too big to store in memory. '
                           'You should either increase mol.max_memory, or set '
                           'mol.incore_anyway. See issue #2473.')




[docs]
def general(eri_or_mol, mo_coeffs, erifile=None, dataname='eri_mo', intor='int2e',
            *args, **kwargs):
    r'''Given four sets of orbitals corresponding to the four MO indices,
    transfer arbitrary spherical AO integrals to MO integrals.

    Args:
        eri_or_mol : ndarray or Mole object
            If AO integrals are given as ndarray, it can be either 8-fold or
            4-fold symmetry.  The integral transformation are computed incore
            (ie all intermediate are held in memory).
            If Mole object is given, AO integrals are generated on the fly and
            outcore algorithm is used (ie intermediate data are held on disk).
        mo_coeffs : 4-item list of ndarray
            Four sets of orbital coefficients, corresponding to the four
            indices of (ij|kl)
        erifile : str or h5py File or h5py Group object
            *Note* this argument is effective when eri_or_mol is Mole object.
            The file to store the transformed integrals.  If not given, the
            transformed integrals are held in memory.

    Kwargs:
        dataname : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        intor : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            *Note* this argument is effective when eri_or_mol is Mole object.
            Permutation symmetry for the AO integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry
            | 'a4ij' : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a4kl' : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | 'a2ij' : anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a2kl' : anti-symmetry between k, l in (ij|kl) (TODO)

        comp : int
            *Note* this argument is effective when eri_or_mol is Mole object.
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        max_memory : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The maximum size of cache to use (in MB), large cache may **not**
            improve performance.
        ioblk_size : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The block size for IO, large block size may **not** improve performance
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four orbital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals

    Returns:
        If eri_or_mol is array or erifile is not give,  the function returns 2D
        array (or 3D array, if comp > 1) of transformed MO integrals.  The MO
        integrals may at most have 4-fold symmetry (if the four sets of orbitals
        are identical) or may not have the permutation symmetry (controlled by
        the kwargs compact).
        Otherwise, return the file/fileobject where the MO integrals are saved.


    Examples:

    >>> from pyscf import gto, ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     with h5py.File(h5file, 'r') as f5:
    ...         print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
    >>> mo2 = numpy.random.random((mol.nao_nr(), 8))
    >>> mo3 = numpy.random.random((mol.nao_nr(), 6))
    >>> mo4 = numpy.random.random((mol.nao_nr(), 4))

    >>> eri1 = ao2mo.general(eri, (mo1,mo2,mo3,mo4))
    >>> print(eri1.shape)
    (80, 24)

    >>> eri1 = ao2mo.general(eri, (mo1,mo2,mo3,mo3))
    >>> print(eri1.shape)
    (80, 21)

    >>> eri1 = ao2mo.general(eri, (mo1,mo2,mo3,mo3), compact=False)
    >>> print(eri1.shape)
    (80, 36)

    >>> eri1 = ao2mo.general(eri, (mo1,mo1,mo2,mo2))
    >>> print(eri1.shape)
    (55, 36)

    >>> eri1 = ao2mo.general(eri, (mo1,mo2,mo1,mo2))
    >>> print(eri1.shape)
    (80, 80)

    >>> ao2mo.general(mol, (mo1,mo2,mo3,mo4), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 24)

    >>> ao2mo.general(mol, (mo1,mo2,mo3,mo3), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 21)

    >>> ao2mo.general(mol, (mo1,mo2,mo3,mo3), 'oh2.h5', compact=False)
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 36)

    >>> ao2mo.general(mol, (mo1,mo1,mo2,mo2), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (55, 36)

    >>> ao2mo.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', dataname='new')
    >>> view('oh2.h5', 'new')
    dataset ['eri_mo', 'new'], shape (55, 55)

    >>> ao2mo.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)

    >>> ao2mo.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    if isinstance(eri_or_mol, numpy.ndarray):
        return incore.general(eri_or_mol, mo_coeffs, *args, **kwargs)
    elif isinstance(eri_or_mol, gto.MoleBase):
        if '_spinor' in intor:
            mod = r_outcore
        else:
            mod = outcore

        if isinstance(erifile, (str, h5py.Group)): # args[0] is erifile
            return mod.general(eri_or_mol, mo_coeffs, erifile, dataname, intor,
                               *args, **kwargs)
        elif isinstance(erifile, tempfile._TemporaryFileWrapper):
            return mod.general(eri_or_mol, mo_coeffs, erifile.name, dataname, intor,
                               *args, **kwargs)
        else:
            return mod.general_iofree(eri_or_mol, mo_coeffs, intor, *args, **kwargs)
    else:
        raise RuntimeError('ERI is not available. If this is generated by mf._eri, '
                           'the integral tensor is too big to store in memory. '
                           'You should either increase mol.max_memory, or set '
                           'mol.incore_anyway. See issue #2473.')




[docs]
def kernel(eri_or_mol, mo_coeffs, erifile=None, dataname='eri_mo', intor='int2e',
           *args, **kwargs):
    r'''Transfer arbitrary spherical AO integrals to MO integrals, for given
    orbitals or four sets of orbitals.  See also :func:`ao2mo.full` and :func:`ao2mo.general`.

    Args:
        eri_or_mol : ndarray or Mole object
            If AO integrals are given as ndarray, it can be either 8-fold or
            4-fold symmetry.  The integral transformation are computed incore
            (ie all intermediate are held in memory).
            If Mole object is given, AO integrals are generated on the fly and
            outcore algorithm is used (ie intermediate data are held on disk).
        mo_coeffs : an np array or a list of arrays
            A matrix of orbital coefficients if it is a numpy ndarray; Or four
            sets of orbital coefficients if it is a list of arrays,
            corresponding to the four indices of (ij|kl).

    Kwargs:
        erifile : str or h5py File or h5py Group object
            *Note* this argument is effective when eri_or_mol is Mole object.
            The file to store the transformed integrals.  If not given, the
            return value is an array (in memory) of the transformed integrals.
        dataname : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        intor : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            *Note* this argument is effective when eri_or_mol is Mole object.
            Permutation symmetry for the AO integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry
            | 'a4ij' : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a4kl' : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | 'a2ij' : anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a2kl' : anti-symmetry between k, l in (ij|kl) (TODO)

        comp : int
            *Note* this argument is effective when eri_or_mol is Mole object.
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        max_memory : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The maximum size of cache to use (in MB), large cache may **not**
            improve performance.
        ioblk_size : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The block size for IO, large block size may **not** improve performance
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four orbital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals

    Returns:
        If eri_or_mol is array or erifile is not give,  the function returns 2D
        array (or 3D array, if comp > 1) of transformed MO integrals.  The MO
        integrals may at most have 4-fold symmetry (if the four sets of orbitals
        are identical) or may not have the permutation symmetry (controlled by
        the kwargs compact).
        Otherwise, return the file/fileobject where the MO integrals are saved.


    Examples:

    >>> from pyscf import gto, ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     with h5py.File(h5file) as f5:
    ...         print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
    >>> mo2 = numpy.random.random((mol.nao_nr(), 8))
    >>> mo3 = numpy.random.random((mol.nao_nr(), 6))
    >>> mo4 = numpy.random.random((mol.nao_nr(), 4))

    >>> eri1 = ao2mo.kernel(mol, mo1)
    >>> print(eri1.shape)
    (55, 55)

    >>> eri = mol.intor('int2e_sph', aosym='s8')
    >>> eri1 = ao2mo.kernel(eri, mo1, compact=False)
    >>> print(eri1.shape)
    (100, 100)

    >>> ao2mo.kernel(mol, mo1, erifile='full.h5')
    >>> view('full.h5')
    dataset ['eri_mo'], shape (55, 55)

    >>> ao2mo.kernel(mol, mo1, 'full.h5', dataname='new', compact=False)
    >>> view('full.h5', 'new')
    dataset ['eri_mo', 'new'], shape (100, 100)

    >>> ao2mo.kernel(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)

    >>> ao2mo.kernel(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)

    >>> eri1 = ao2mo.kernel(eri, (mo1,mo2,mo3,mo4))
    >>> print(eri1.shape)
    (80, 24)

    >>> eri1 = ao2mo.kernel(eri, (mo1,mo2,mo3,mo3))
    >>> print(eri1.shape)
    (80, 21)

    >>> eri1 = ao2mo.kernel(eri, (mo1,mo2,mo3,mo3), compact=False)
    >>> print(eri1.shape)
    (80, 36)

    >>> eri1 = ao2mo.kernel(eri, (mo1,mo1,mo2,mo2))
    >>> print(eri1.shape)
    (55, 36)

    >>> eri1 = ao2mo.kernel(eri, (mo1,mo2,mo1,mo2))
    >>> print(eri1.shape)
    (80, 80)

    >>> ao2mo.kernel(mol, (mo1,mo2,mo3,mo4), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 24)

    >>> ao2mo.kernel(mol, (mo1,mo2,mo3,mo3), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 21)

    >>> ao2mo.kernel(mol, (mo1,mo2,mo3,mo3), 'oh2.h5', compact=False)
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 36)

    >>> ao2mo.kernel(mol, (mo1,mo1,mo2,mo2), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (55, 36)

    >>> ao2mo.kernel(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', dataname='new')
    >>> view('oh2.h5', 'new')
    dataset ['eri_mo', 'new'], shape (55, 55)

    >>> ao2mo.kernel(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)

    >>> ao2mo.kernel(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        return full(eri_or_mol, mo_coeffs, erifile, dataname, intor, *args, **kwargs)
    else:
        return general(eri_or_mol, mo_coeffs, erifile, dataname, intor, *args, **kwargs)




[docs]
def get_ao_eri(mol):
    '''2-electron integrals in AO basis'''
    return mol.intor('int2e', aosym='s4')



get_mo_eri = kernel

```