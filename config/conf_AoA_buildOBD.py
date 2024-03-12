import os
from os.path import join as opj

import numpy as np
import delensalot
from delensalot.config.metamodel.dlensalot_mm import *

fg = '11'
ai = 2
desc_flag = ''

if ai == 1:
    nlevp = 0.35
elif ai == 2:
    nlevp = 0.41
elif ai == 3:
    nlevp = 1.30

maskrhits_dir = '/pscratch/sd/s/sebibel/analysis/OBDmatrix/AoA/a{ai}lat.{fg}/lcut200/'.format(ai=ai, fg=fg)
rtreshold = 30
mask_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'AoA', 'a{ai}lat.{fg}'.format(ai=ai, fg=fg), 'lcut200', 'mask_tresh{}.fits'.format(rtreshold))
rhits_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'AoA', 'a{ai}lat.{fg}'.format(ai=ai, fg=fg), 'lcut200', 'rhits_tresh{}.fits'.format(rtreshold))

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["build_OBD"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 32
    ),
    analysis = DLENSALOT_Analysis(
        mask = mask_fn,
        lmin_teb = (70, 70, 200)
    ),
    noisemodel = DLENSALOT_Noisemodel(
        OBD = 'OBD',
        sky_coverage = 'masked',
        spectrum_type = 'white',
        nlev = {'P': nlevp, 'T': np.sqrt(1)},
        rhits_normalised = (rhits_fn, np.inf)
    ),
    obd = DLENSALOT_OBD(
        libdir = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'AoA', 'a{ai}'.format(ai=ai), 'lcut200',),
        nlev_dep = 1e4,
    )
)