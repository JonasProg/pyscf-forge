#!/usr/bin/env python
# Copyright 2019-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Analytical Hessian for Embedding
'''

import numpy as np

try:
    import pyframe
except ImportError:
    raise ImportError(
        'Unable to import PyFraME. Please install PyFraME.')

from pyframe.embedding import (electrostatic_interactions, induction_interactions, repulsion_interactions,
                               dispersion_interactions)

from pyscf.lib import logger
from pyscf import gto
from pyscf import df, scf
from pyscf.embedding._attach_embedding import _Embedding

from pyscf.hessian import rhf as rhf_hess


def make_hess_object(hess_method):
    '''
    '''
    assert isinstance(hess_method.base, _Embedding)
    if not isinstance(hess_method.base, scf.hf.SCF):
        raise NotImplementedError("PE gradients only implemented for SCF methods.")
    hess_method_class = hess_method.__class__

    class EmbeddingHess(hess_method_class):
        def __init__(self, grad_method):
            self.__dict__.update(grad_method.__dict__)
            self.de_classical_subsystem = None
            self.de_quantum_subsystem = None
            self._keys = self._keys.union(['de_classical_subsystem', 'de_quantum_subsystem'])

        def kernel(self, *args, dm=None, atmlst=None, **kwargs):
            '''
            '''
            dm = kwargs.pop('dm', None)
            if dm is None:
                dm = self.base.make_rdm1(ao_repr=True)

            self.de_classical_subsystem = kernel(self.base.with_embedding, dm)
            self.de_quantum_subsystem = hess_method_class.kernel(self, *args, **kwargs)
            self.de = self.de_quantum_subsystem + self.de_classical_subsystem

            if self.verbose >= logger.NOTE:
                logger.note(self, '--------------- %s (+%s) gradients ---------------',
                            self.base.__class__.__name__,
                            self.base.with_embedding.__class__.__name__)
                rhf_hess._write(self, self.mol, self.de, self.atmlst)
                logger.note(self, '----------------------------------------------')
            return self.de

        def _finalize(self):
            # disable _finalize. It is called in grad_method.kernel method
            # where self.de was not yet initialized.
            pass

    return EmbeddingHess(hess_method_class)

def kernel(embedding_obj, dm, verbose=None):
    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]

    mol = embedding_obj.mol
    natoms = mol.natm
    de = np.zeros((3 * natoms, 3 * natoms))
    quantum_subsystem = embedding_obj.quantum_subsystem
    classical_subsystem = embedding_obj.classical_subsystem
    # give actual contributions
    de = 0
    return de


