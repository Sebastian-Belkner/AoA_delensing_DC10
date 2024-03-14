"""
Masked sky iterative delensing on simulated CMB-S4 AoA observed CMB polarization data.

QE and iterative reconstruction uses anisotropic filters. 
"""

import numpy as np
import os
from os.path import join as opj
import healpy as hp

import delensalot
from delensalot import utils
from delensalot.utility.utils_hp import gauss_beam
import delensalot.core.power.pospace as pospace
from delensalot.config.config_helper import LEREPI_Constants as lc
from delensalot.config.metamodel.dlensalot_mm import *

fg = '12'
ai = 3
desc_flag = 'mfsimsub'
rtreshold_delens = 1.3

maskrhits_dir = '/pscratch/sd/s/sebibel/analysis/OBDmatrix/AoA/a{ai}lat.{fg}/lcut200/'.format(ai=ai, fg=fg)
misc_dir = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/AoA')
data_dir = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps')
rtreshold = 30
mask_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'AoA', 'a{ai}lat.{fg}'.format(ai=ai, fg=fg), 'lcut200', 'mask_tresh{}.fits'.format(rtreshold))
rhits_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'AoA', 'a{ai}lat.{fg}'.format(ai=ai, fg=fg), 'lcut200', 'rhits_tresh{}.fits'.format(rtreshold))
rhits_nonorm_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'AoA', 'a{ai}lat.{fg}'.format(ai=ai, fg=fg), 'lcut200', 'rhits_tresh{}_nonorm.fits'.format(rtreshold))

misc_dir = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/AoA')
data_dir = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps')

def func(data):
    # print('======================')
    # print('applying func to data')
    # print('======================')
    return data #* np.nan_to_num(utils.cli(hp.read_map(rhits_nonorm_fn)))


dlensalot_model = DLENSALOT_Model(
    defaults_to = 'default_CMBS4_maskedsky_polarization',
    validate_model = False,
    job = DLENSALOT_Job(
        jobs = ["QE_lensrec", "MAP_lensrec", "delens"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 8
    ),                              
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        version = '',
        simidxs = np.arange(0,5),
        simidxs_mf = np.arange(0,0),
        TEMP_suffix = 'AoA_a{}_fg{}_{}'.format(ai,fg, desc_flag),
        Lmin = 3, 
        lm_max_ivf = (4000, 4000),
        lmin_teb = (70, 70, 200),
        # zbounds = ('mr_relative', 10.),
        # zbounds_len = ('extend', 5.),
        zbounds = (-1,1),
        zbounds_len = (-1,1),
        beam = 2.5,
        mask = mask_fn
    ),
    simulationdata = DLENSALOT_Simulation(
        space = 'map', 
        flavour = 'obs',
        lmax = 4096,
        phi_lmax = 5120,
        spin = 0,
        libdir = opj(data_dir, '10a{ai}lat.{fg}/'.format(ai=ai, fg=fg)),
        libdir_noise = opj(data_dir, '10a{ai}lat.{fg}/'.format(ai=ai, fg=fg)),
        fnsnoise = {
            'E':'cmb-s4_nilc_EBresidual-noise_a{ai}latp{fg}_beam02.50_ellmin70_2048_mc{{:03d}}.fits'.format(ai=ai, fg=fg),
            'B':'cmb-s4_nilc_EBresidual-noise_a{ai}latp{fg}_beam02.50_ellmin70_2048_mc{{:03d}}.fits'.format(ai=ai, fg=fg)
        },
        fns = {
            'E':'cmb-s4_nilc_EBmap_a{ai}latp{fg}_beam02.50_ellmin70_2048_mc{{:03d}}.fits'.format(ai=ai, fg=fg),
            'B':'cmb-s4_nilc_EBmap_a{ai}latp{fg}_beam02.50_ellmin70_2048_mc{{:03d}}.fits'.format(ai=ai, fg=fg)
        },
        CMB_modifier = func,
        transfunction = gauss_beam(2.5/180/60 * np.pi, lmax=4096),
        geominfo = ('healpix', {'nside': 2048}),
    ),
    noisemodel = DLENSALOT_Noisemodel(
        OBD = 'OBD',
        sky_coverage = 'masked',
        spectrum_type = 'white',
        nlev = {'P': 1.30, 'T': np.sqrt(1)},
        rhits_normalised = (rhits_fn, np.inf)
    ),
    obd = DLENSALOT_OBD(
        libdir = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'AoA', 'a{ai}'.format(ai=ai), 'lcut200',),
        nlev_dep = 1e4,
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_blt"],
        filter_directional = 'anisotropic',
        lm_max_qlm = (4000, 4000),
        cg_tol = 1e-4
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_blt"],
        filter_directional = 'anisotropic',
        itmax = 10,
        cg_tol = 1e-4,
        lm_max_unl = (4200, 4200),
        lm_max_qlm = (4000, 4000),
        stepper = DLENSALOT_Stepper(
            typ = 'harmonicbump',
            lmax_qlm = 4000,
            mmax_qlm = 4000,
            a = 0.5,
            b = 0.499,
            xa = 400,
            xb = 1500
        ),
        mfvar = '/pscratch/sd/s/sebibel/analysis/AoA_a{ai}_mf_MFsim_nlev05_OBD/QE/mf_allsims.npy'.format(ai=ai)
    ),
    madel = DLENSALOT_Mapdelensing(
        data_from_CFS = False,
        edges = lc.AoA_edges,
        iterations = [10],
        masks_fn = ['/pscratch/sd/s/sebibel/analysis/OBDmatrix/AoA/a{ai}lat.{fg}/lcut200/mask_tresh{rtreshold_delens}.fits'.format(ai=ai, fg=fg, rtreshold_delens=rtreshold_delens)],
        lmax = 1024,
        Cl_fid = 'obs', #this doesn't make sense right now..
        basemap = 'obs',
        libdir_it = None,
        binning = 'binned',
        spectrum_calculator = pospace,
    )
)