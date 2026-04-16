#!/usr/bin/env python3
"""Batch generate task bank JSON files for all 17 missing real-data systems.

Reads raw RV data from downloads/ (NASA ExoArchive .tbl and VizieR .tsv),
combines with ground-truth planet parameters from the literature,
and writes task bank JSON files to stargazer/stargazer_bank/.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from dataclasses import asdict

import numpy as np

from stargazer.config import (
    Task, Observations, SystemConfig, StarParams, PlanetParams,
    InstrumentParams, ObservingSchedule, NoiseParams, GPParams,
)

BANK_DIR = Path("stargazer/stargazer_bank")
DOWNLOADS = Path("downloads")
TWO_PI = 2.0 * math.pi


# =====================================================================
# Planet parameters from literature
# =====================================================================
# Each entry: task_id -> dict with star info + list of planets
# Planet fields: P_days, m_sin_i_mjup, e, omega_deg, K_ms,
#   and one of T_peri_jd / T_conj_jd for phase computation.
#   If neither epoch is known, l_rad is fit from data.

SYSTEMS = {
    # ---- No.4  HD 179949  (Diff 2) ----
    # Butler et al. 2006, Tinney et al. 2001
    "HD179949_Butler2006": {
        "target_name": "HD 179949",
        "M_star_sun": 1.24,
        "spectral_type": "F8V",
        "difficulty": 2,
        "reference": "Butler et al. 2006 (ApJ, 646, 505)",
        "data_source": "nasa_exoarchive",
        "data_dir": "row4_HD_179949_b",
        "use_ref": "Butler et al. 2006 (ApJ, 646, 505)",  # pick the largest dataset
        "planets": [
            {
                "P_days": 3.0925,
                "m_sin_i_mjup": 0.916,
                "e": 0.022,
                "omega_deg": 192.0,
                "K_ms": 112.6,
                "T_peri_jd": 2451001.51,
            },
        ],
        "challenge": "hot Jupiter, multi-instrument, simple signal",
    },

    # ---- No.5  HD 217107  (Diff 3) ----
    # Wright et al. 2009
    "HD217107_Wright2009": {
        "target_name": "HD 217107",
        "M_star_sun": 1.02,
        "spectral_type": "G8IV",
        "difficulty": 3,
        "reference": "Wright et al. 2009 (ApJ, 693, 1084)",
        "data_source": "nasa_exoarchive",
        "data_dir": "row5_HD_217107_b",
        "use_ref": "Wright et al. 2009 (ApJ, 693, 1084)",
        "planets": [
            {
                "P_days": 7.12689,
                "m_sin_i_mjup": 1.33,
                "e": 0.132,
                "omega_deg": 22.7,
                "K_ms": 139.7,
                "T_peri_jd": 2450505.61,
            },
        ],
        "challenge": "1 planet, moderate eccentricity, multi-instrument",
    },

    # ---- No.6  HD 88133  (Diff 3) ----
    # Butler et al. 2006
    "HD88133_Butler2006": {
        "target_name": "HD 88133",
        "M_star_sun": 1.20,
        "spectral_type": "G5IV",
        "difficulty": 3,
        "reference": "Butler et al. 2006 (ApJ, 646, 505)",
        "data_source": "nasa_exoarchive",
        "data_dir": "row6_HD_88133_b",
        "use_ref": "Butler et al. 2006 (ApJ, 646, 505)",
        "planets": [
            {
                "P_days": 3.4157,
                "m_sin_i_mjup": 0.22,
                "e": 0.133,
                "omega_deg": 349.0,
                "K_ms": 35.7,
                "T_peri_jd": 2453023.5,
            },
        ],
        "challenge": "1 planet, moderate SNR, sparse observations",
    },

    # ---- No.7  HD 12661  (Diff 4) ----
    # Wright et al. 2009
    "HD12661_Wright2009": {
        "target_name": "HD 12661",
        "M_star_sun": 1.07,
        "spectral_type": "G6V",
        "difficulty": 4,
        "reference": "Wright et al. 2009 (ApJ, 693, 1084)",
        "data_source": "nasa_exoarchive",
        "data_dir": "row7_HD_12661_b",
        "use_ref": "Wright et al. 2009 (ApJ, 693, 1084)",
        "planets": [
            {
                "P_days": 263.6,
                "m_sin_i_mjup": 2.30,
                "e": 0.350,
                "omega_deg": 296.3,
                "K_ms": 74.4,
                "T_peri_jd": 2450672.9,
            },
            {
                "P_days": 1444.5,
                "m_sin_i_mjup": 1.57,
                "e": 0.20,
                "omega_deg": 165.7,
                "K_ms": 27.6,
                "T_peri_jd": 2451353.9,
            },
        ],
        "challenge": "2 planets, clear period separation, multi-instrument",
    },

    # ---- No.8  HD 37124  (Diff 4) ----
    # Vogt et al. 2005
    "HD37124_Vogt2005": {
        "target_name": "HD 37124",
        "M_star_sun": 0.85,
        "spectral_type": "G4V",
        "difficulty": 4,
        "reference": "Vogt et al. 2005 (ApJ, 632, 638)",
        "data_source": "nasa_exoarchive",
        "data_dir": "row8_HD_37124_b",
        "use_ref": "Vogt et al. 2005 (ApJ, 632, 638)",
        "planets": [
            {
                "P_days": 154.46,
                "m_sin_i_mjup": 0.61,
                "e": 0.054,
                "omega_deg": 140.5,
                "K_ms": 28.5,
                "T_peri_jd": 2451225.0,
            },
            {
                "P_days": 2295.0,
                "m_sin_i_mjup": 0.60,
                "e": 0.14,
                "omega_deg": 266.0,
                "K_ms": 12.2,
                "T_peri_jd": 2451675.0,
            },
        ],
        "challenge": "2 planets, long-period outer planet, single instrument",
    },

    # ---- No.9  HD 69830  (Diff 5) ----
    # Lovis et al. 2006, Nature 441, 305
    "HD69830_Lovis2006": {
        "target_name": "HD 69830",
        "M_star_sun": 0.86,
        "spectral_type": "K0V",
        "difficulty": 5,
        "reference": "Lovis et al. 2006 (Nature, 441, 305)",
        "data_source": "nasa_exoarchive",
        "data_dir": "row9_HD_69830_b",
        "use_ref": "Lovis et al. 2006 (Nature, 441, 305)",
        "planets": [
            {
                "P_days": 8.667,
                "m_sin_i_mjup": 0.0328,
                "e": 0.10,
                "omega_deg": 340.0,
                "K_ms": 3.51,
                "T_peri_jd": 2453496.8,
            },
            {
                "P_days": 31.56,
                "m_sin_i_mjup": 0.0372,
                "e": 0.13,
                "omega_deg": 221.0,
                "K_ms": 2.66,
                "T_peri_jd": 2453469.3,
            },
            {
                "P_days": 197.0,
                "m_sin_i_mjup": 0.057,
                "e": 0.07,
                "omega_deg": 224.0,
                "K_ms": 2.20,
                "T_peri_jd": 2453472.0,
            },
        ],
        "challenge": "3 low-mass planets, all K < 4 m/s, classic super-Earth system",
    },

    # ---- No.10  HD 74156  (Diff 5) ----
    # Naef et al. 2004
    "HD74156_Naef2004": {
        "target_name": "HD 74156",
        "M_star_sun": 1.24,
        "spectral_type": "G0",
        "difficulty": 5,
        "reference": "Naef et al. 2004 (A&A, 414, 351)",
        "data_source": "vizier",
        "data_file": "HD_74156__J_A+A_414_351_table2.tsv",
        "rv_unit": "km/s",
        "has_instr_col": True,
        "instr_map": {"1": "ELODIE", "2": "CORALIE"},
        "planets": [
            {
                "P_days": 51.638,
                "m_sin_i_mjup": 1.86,
                "e": 0.636,
                "omega_deg": 181.5,
                "K_ms": 112.0,
                "T_peri_jd": 2451981.3,
            },
            {
                "P_days": 2476.0,
                "m_sin_i_mjup": 8.03,
                "e": 0.583,
                "omega_deg": 242.4,
                "K_ms": 125.0,
                "T_peri_jd": 2451824.0,
            },
        ],
        "challenge": "2 planets, high eccentricities, multi-instrument",
    },

    # ---- No.11  HD 40307  (Diff 6) ----
    # Mayor et al. 2009
    "HD40307_Mayor2009": {
        "target_name": "HD 40307",
        "M_star_sun": 0.77,
        "spectral_type": "K2.5V",
        "difficulty": 6,
        "reference": "Mayor et al. 2009 (A&A, 493, 639)",
        "data_source": "vizier",
        "data_file": "HD_40307__J_A+A_493_639_table1.tsv",
        "rv_unit": "km/s",
        "has_instr_col": False,
        "default_instrument": "HARPS",
        "planets": [
            {
                "P_days": 4.3115,
                "m_sin_i_mjup": 0.0132,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 1.94,
                "T_conj_jd": 2454562.47,
            },
            {
                "P_days": 9.620,
                "m_sin_i_mjup": 0.0211,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 2.47,
                "T_conj_jd": 2454555.87,
            },
            {
                "P_days": 20.46,
                "m_sin_i_mjup": 0.0283,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 2.54,
                "T_conj_jd": 2454558.80,
            },
        ],
        "challenge": "3 super-Earths, close periods, all K < 3 m/s",
    },

    # ---- No.12  HD 10180  (Diff 6) ----
    # Lovis et al. 2011
    "HD10180_Lovis2011": {
        "target_name": "HD 10180",
        "M_star_sun": 1.06,
        "spectral_type": "G1V",
        "difficulty": 6,
        "reference": "Lovis et al. 2011 (A&A, 528, A112)",
        "data_source": "vizier",
        "data_file": "HD_10180__J_A+A_528_A112_table1.tsv",
        "rv_unit": "km/s",
        "has_instr_col": False,
        "default_instrument": "HARPS",
        "planets": [
            # Focus on the 5 most secure planets (c,d,e,f,g)
            {
                "P_days": 5.7596,
                "m_sin_i_mjup": 0.0412,
                "e": 0.045,
                "omega_deg": 322.0,
                "K_ms": 4.54,
                "T_peri_jd": 2452942.0,
            },
            {
                "P_days": 16.358,
                "m_sin_i_mjup": 0.0378,
                "e": 0.088,
                "omega_deg": 316.0,
                "K_ms": 2.93,
                "T_peri_jd": 2452947.0,
            },
            {
                "P_days": 49.745,
                "m_sin_i_mjup": 0.0805,
                "e": 0.026,
                "omega_deg": 146.0,
                "K_ms": 4.15,
                "T_peri_jd": 2452985.0,
            },
            {
                "P_days": 122.72,
                "m_sin_i_mjup": 0.0723,
                "e": 0.135,
                "omega_deg": 137.0,
                "K_ms": 2.80,
                "T_peri_jd": 2453009.0,
            },
            {
                "P_days": 601.2,
                "m_sin_i_mjup": 0.203,
                "e": 0.19,
                "omega_deg": 277.0,
                "K_ms": 3.46,
                "T_peri_jd": 2453137.0,
            },
        ],
        "challenge": "5+ planets, low K, close periods, HARPS precision",
    },

    # ---- No.13  HD 20794  (Diff 7) ----
    # Pepe et al. 2011
    "HD20794_Pepe2011": {
        "target_name": "HD 20794",
        "M_star_sun": 0.70,
        "spectral_type": "G8V",
        "difficulty": 7,
        "reference": "Pepe et al. 2011 (A&A, 534, A58)",
        "data_source": "vizier",
        "data_file": "HD_20794__J_A+A_534_A58_table1.tsv",
        "rv_unit": "km/s",
        "has_instr_col": False,
        "default_instrument": "HARPS",
        "planets": [
            {
                "P_days": 18.315,
                "m_sin_i_mjup": 0.0085,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 0.83,
                "T_conj_jd": 2453741.0,
            },
            {
                "P_days": 40.114,
                "m_sin_i_mjup": 0.0115,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 0.56,
                # fit from data
            },
            {
                "P_days": 90.309,
                "m_sin_i_mjup": 0.0148,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 0.85,
                # fit from data
            },
        ],
        "challenge": "3 super-Earths, sub-m/s signals, extremely low K",
    },

    # ---- No.14  HD 215152  (Diff 7) ----
    # Delisle et al. 2018
    "HD215152_Delisle2018": {
        "target_name": "HD 215152",
        "M_star_sun": 0.77,
        "spectral_type": "K3V",
        "difficulty": 7,
        "reference": "Delisle et al. 2018 (A&A, 614, A133)",
        "data_source": "vizier",
        "data_files": [
            "HD_215152__J_A+A_614_A133_harps_a.tsv",
            "HD_215152__J_A+A_614_A133_harps_b.tsv",
        ],
        "rv_unit": "km/s",
        "has_instr_col": False,
        "multi_file_instruments": ["HARPS_pre", "HARPS_post"],
        "bjd_offset": 2400000.0,  # BJD in file is BJD-2400000
        "planets": [
            {
                "P_days": 5.760,
                "m_sin_i_mjup": 0.006,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 0.80,
                # fit from data
            },
            {
                "P_days": 7.282,
                "m_sin_i_mjup": 0.007,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 0.87,
                # fit from data
            },
        ],
        "challenge": "2 super-Earths, sub-m/s signals, HARPS pre/post fiber upgrade offset",
    },

    # ---- No.15  GJ 876  (Diff 8) ----
    # Rivera et al. 2010
    "GJ876_Rivera2010": {
        "target_name": "GJ 876",
        "M_star_sun": 0.334,
        "spectral_type": "M4V",
        "difficulty": 8,
        "reference": "Rivera et al. 2010 (ApJ, 719, 890)",
        "data_source": "vizier",
        "data_file": "GJ_876__J_ApJ_719_890_table1.tsv",
        "rv_unit": "m/s",
        "has_instr_col": False,
        "default_instrument": "HIRES",
        "bjd_offset": 2450000.0,  # BJD in file is BJD-2450000
        "planets": [
            # 4-planet system; b, c in 2:1 resonance
            {
                "P_days": 61.1166,
                "m_sin_i_mjup": 2.2756,
                "e": 0.0324,
                "omega_deg": 50.3,
                "K_ms": 214.00,
                "T_peri_jd": 2452490.3,
            },
            {
                "P_days": 30.0881,
                "m_sin_i_mjup": 0.7142,
                "e": 0.2243,
                "omega_deg": 48.76,
                "K_ms": 88.36,
                "T_peri_jd": 2452467.5,
            },
            {
                "P_days": 1.93778,
                "m_sin_i_mjup": 0.0208,
                "e": 0.207,
                "omega_deg": 234.0,
                "K_ms": 6.56,
                "T_peri_jd": 2452366.7,
            },
            {
                "P_days": 124.26,
                "m_sin_i_mjup": 0.046,
                "e": 0.055,
                "omega_deg": 0.0,
                "K_ms": 3.49,
                "T_peri_jd": 2452440.0,
            },
        ],
        "challenge": "4 planets, Laplace resonance (b:c = 2:1), M-dwarf, large K range",
    },

    # ---- No.16  HD 45364  (Diff 8) ----
    # Correia et al. 2009
    "HD45364_Correia2009": {
        "target_name": "HD 45364",
        "M_star_sun": 0.82,
        "spectral_type": "K0V",
        "difficulty": 8,
        "reference": "Correia et al. 2009 (A&A, 496, 521)",
        "data_source": "vizier",
        "data_file": "HD_45364__J_A+A_496_521_hd45364.tsv",
        "rv_unit": "km/s",
        "has_instr_col": False,
        "default_instrument": "CORALIE",
        "planets": [
            {
                "P_days": 226.93,
                "m_sin_i_mjup": 0.1872,
                "e": 0.1684,
                "omega_deg": 162.6,
                "K_ms": 7.22,
                "T_peri_jd": 2453937.0,
            },
            {
                "P_days": 342.85,
                "m_sin_i_mjup": 0.6579,
                "e": 0.0974,
                "omega_deg": 7.41,
                "K_ms": 21.22,
                "T_peri_jd": 2453974.5,
            },
        ],
        "challenge": "2 planets in 3:2 resonance, moderate eccentricities",
    },

    # ---- No.17  55 Cancri  (Diff 9) ----
    # Naef et al. 2004 (ELODIE/CORALIE data)
    "55Cnc_Naef2004": {
        "target_name": "55 Cancri",
        "M_star_sun": 0.95,
        "spectral_type": "G8V",
        "difficulty": 9,
        "reference": "Naef et al. 2004 (A&A, 414, 351)",
        "data_source": "vizier",
        "data_file": "55_Cancri__J_A+A_414_351_table3.tsv",
        "rv_unit": "km/s",
        "has_instr_col": True,
        "instr_map": {"1": "ELODIE", "2": "CORALIE"},
        "planets": [
            # b (confirmed)
            {
                "P_days": 14.651,
                "m_sin_i_mjup": 0.824,
                "e": 0.014,
                "omega_deg": 131.0,
                "K_ms": 71.3,
                "T_peri_jd": 2449990.0,
            },
            # c (long period)
            {
                "P_days": 5218.0,
                "m_sin_i_mjup": 3.835,
                "e": 0.063,
                "omega_deg": 77.0,
                "K_ms": 45.2,
                "T_peri_jd": 2451000.0,
            },
        ],
        "challenge": "multi-planet, complex architecture, ELODIE/CORALIE data",
    },

    # ---- No.18  HD 160691 / mu Ara  (Diff 9) ----
    # Benedict et al. 2022
    "HD160691_Benedict2022": {
        "target_name": "HD 160691 (mu Ara)",
        "M_star_sun": 1.08,
        "spectral_type": "G3IV-V",
        "difficulty": 9,
        "reference": "Benedict et al. 2022 (AJ, 163, 295)",
        "data_source": "vizier",
        "data_file": "HD_160691_mu_Ara__J_AJ_163_295_table2.tsv",
        "rv_unit": "km/s",
        "has_instr_col": False,
        "has_ref_col": True,
        "ref_instrument_map": {
            "11": "HARPS",
            "12": "HARPS",
            "14": "AAT",
            "15": "Magellan",
        },
        "bjd_offset": 2400000.0,
        "planets": [
            # c (the "hot Neptune" / "hot super-Earth")
            {
                "P_days": 9.6386,
                "m_sin_i_mjup": 0.033,
                "e": 0.172,
                "omega_deg": 213.0,
                "K_ms": 3.06,
                "T_peri_jd": 2453009.0,
            },
            # d (inner giant)
            {
                "P_days": 310.55,
                "m_sin_i_mjup": 0.521,
                "e": 0.0666,
                "omega_deg": 189.6,
                "K_ms": 14.91,
                "T_peri_jd": 2452708.0,
            },
            # b (outer giant)
            {
                "P_days": 643.25,
                "m_sin_i_mjup": 1.676,
                "e": 0.128,
                "omega_deg": 22.0,
                "K_ms": 37.78,
                "T_peri_jd": 2452365.0,
            },
            # e (outermost)
            {
                "P_days": 4205.8,
                "m_sin_i_mjup": 1.814,
                "e": 0.098,
                "omega_deg": 57.6,
                "K_ms": 14.06,
                "T_peri_jd": 2453300.0,
            },
        ],
        "challenge": "4 planets, wide K range, multi-instrument compilation, complex",
    },

    # ---- No.19  HD 10180 full  (Diff 10) ----
    # Same data as No.12 but difficulty 10 (expect all 6-7 planets)
    "HD10180_full_Lovis2011": {
        "target_name": "HD 10180",
        "M_star_sun": 1.06,
        "spectral_type": "G1V",
        "difficulty": 10,
        "reference": "Lovis et al. 2011 (A&A, 528, A112)",
        "data_source": "vizier",
        "data_file": "HD_10180__J_A+A_528_A112_table1.tsv",
        "rv_unit": "km/s",
        "has_instr_col": False,
        "default_instrument": "HARPS",
        "planets": [
            # All 7 proposed planets (b,c,d,e,f,g,h)
            {
                "P_days": 1.17768,
                "m_sin_i_mjup": 0.0044,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 0.78,
                # fit from data
            },
            {
                "P_days": 5.7596,
                "m_sin_i_mjup": 0.0412,
                "e": 0.045,
                "omega_deg": 322.0,
                "K_ms": 4.54,
                "T_peri_jd": 2452942.0,
            },
            {
                "P_days": 16.358,
                "m_sin_i_mjup": 0.0378,
                "e": 0.088,
                "omega_deg": 316.0,
                "K_ms": 2.93,
                "T_peri_jd": 2452947.0,
            },
            {
                "P_days": 49.745,
                "m_sin_i_mjup": 0.0805,
                "e": 0.026,
                "omega_deg": 146.0,
                "K_ms": 4.15,
                "T_peri_jd": 2452985.0,
            },
            {
                "P_days": 122.72,
                "m_sin_i_mjup": 0.0723,
                "e": 0.135,
                "omega_deg": 137.0,
                "K_ms": 2.80,
                "T_peri_jd": 2453009.0,
            },
            {
                "P_days": 601.2,
                "m_sin_i_mjup": 0.203,
                "e": 0.19,
                "omega_deg": 277.0,
                "K_ms": 3.46,
                "T_peri_jd": 2453137.0,
            },
            {
                "P_days": 2222.0,
                "m_sin_i_mjup": 0.203,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 1.27,
                # fit from data
            },
        ],
        "challenge": "7 planets including sub-m/s signals, full complexity",
    },

    # ---- No.20  GJ 581  (Diff 10) ----
    # Mayor et al. 2009
    "GJ581_Mayor2009": {
        "target_name": "GJ 581",
        "M_star_sun": 0.31,
        "spectral_type": "M3V",
        "difficulty": 10,
        "reference": "Mayor et al. 2009 (A&A, 507, 487)",
        "data_source": "vizier",
        "data_file": "GJ_581__J_A+A_507_487_table1.tsv",
        "rv_unit": "km/s",
        "has_instr_col": False,
        "default_instrument": "HARPS",
        "planets": [
            # b
            {
                "P_days": 5.3687,
                "m_sin_i_mjup": 0.0492,
                "e": 0.031,
                "omega_deg": 251.0,
                "K_ms": 12.45,
                "T_peri_jd": 2454752.0,
            },
            # c
            {
                "P_days": 12.929,
                "m_sin_i_mjup": 0.0169,
                "e": 0.17,
                "omega_deg": 228.0,
                "K_ms": 3.18,
                "T_peri_jd": 2454759.0,
            },
            # d
            {
                "P_days": 66.87,
                "m_sin_i_mjup": 0.022,
                "e": 0.38,
                "omega_deg": 271.0,
                "K_ms": 2.53,
                "T_peri_jd": 2454790.0,
            },
            # e
            {
                "P_days": 3.1490,
                "m_sin_i_mjup": 0.006,
                "e": 0.0,
                "omega_deg": 0.0,
                "K_ms": 1.85,
                # fit from data
            },
        ],
        "challenge": "4 planets, M-dwarf, stellar activity, low-K signals, contested planets",
    },
}


# =====================================================================
# Data parsers
# =====================================================================

def parse_nasa_tbl(filepath: Path) -> tuple:
    """Parse a NASA ExoArchive .tbl file. Returns (times, rvs, sigmas, instrument)."""
    times, rvs, sigmas = [], [], []
    instrument = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("\\INSTRUMENT"):
                instrument = line.split("=")[1].strip().strip('"')
            if line.startswith("|") or line.startswith("\\") or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    t = float(parts[0])
                    rv = float(parts[1])
                    sig = float(parts[2])
                    times.append(t)
                    rvs.append(rv)
                    sigmas.append(sig)
                except ValueError:
                    continue
    return np.array(times), np.array(rvs), np.array(sigmas), instrument or "UNKNOWN"


def parse_vizier_tsv(filepath: Path, rv_unit: str = "km/s",
                     has_instr_col: bool = False,
                     instr_map: dict = None,
                     default_instrument: str = "UNKNOWN",
                     bjd_offset: float = 0.0,
                     has_ref_col: bool = False,
                     ref_instrument_map: dict = None,
                     ) -> tuple:
    """Parse a VizieR TSV file. Returns (times, rvs_ms, sigmas_ms, instruments)."""
    times, rvs, sigmas, instruments = [], [], [], []
    rv_scale = 1000.0 if rv_unit == "km/s" else 1.0

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Skip header rows (column names, units, separator)
            parts = line.split("\t")
            if not parts:
                continue

            # Detect header lines
            first = parts[0].strip()
            if first in ("BJD", "d", "Ref", "JD") or first.startswith("-"):
                continue

            try:
                if has_ref_col and ref_instrument_map:
                    # Format: Ref  JD  RVel  e_RVel
                    ref_id = parts[0].strip()
                    t = float(parts[1].strip())
                    rv = float(parts[2].strip())
                    sig = float(parts[3].strip())
                    inst = ref_instrument_map.get(ref_id, f"Ref{ref_id}")
                elif has_instr_col:
                    # Format: BJD  RV  e_RV  Instr
                    t = float(parts[0].strip())
                    rv = float(parts[1].strip())
                    sig = float(parts[2].strip())
                    instr_code = parts[3].strip()
                    inst = (instr_map or {}).get(instr_code, f"Inst{instr_code}")
                else:
                    # Format: BJD  RV  e_RV  [optional extra cols]
                    t = float(parts[0].strip())
                    rv = float(parts[1].strip())
                    sig = float(parts[2].strip())
                    inst = default_instrument

                if bjd_offset > 0:
                    t += bjd_offset

                times.append(t)
                rvs.append(rv * rv_scale)
                sigmas.append(sig * rv_scale)
                if not (has_ref_col and ref_instrument_map):
                    instruments.append(inst if has_instr_col else default_instrument)
                else:
                    instruments.append(inst)
            except (ValueError, IndexError):
                continue

    return np.array(times), np.array(rvs), np.array(sigmas), instruments


def load_nasa_data(sys_info: dict) -> tuple:
    """Load data from NASA ExoArchive, selecting the reference dataset."""
    data_dir = DOWNLOADS / "nasa_exoarchive" / sys_info["data_dir"]
    use_ref = sys_info.get("use_ref", "")

    all_times, all_rvs, all_sigmas, all_insts = [], [], [], []

    for tbl_file in sorted(data_dir.glob("*.tbl")):
        times, rvs, sigmas, inst = parse_nasa_tbl(tbl_file)
        if len(times) == 0:
            continue

        # Check if this file's reference matches
        with open(tbl_file) as f:
            content = f.read()
        ref_match = re.search(r'\\REFERENCE\s*=\s*"(.+?)"', content)
        file_ref = ref_match.group(1) if ref_match else ""

        if use_ref and file_ref != use_ref:
            continue

        all_times.extend(times.tolist())
        all_rvs.extend(rvs.tolist())
        all_sigmas.extend(sigmas.tolist())
        all_insts.extend([inst] * len(times))

    # Sort by time
    idx = np.argsort(all_times)
    times = np.array(all_times)[idx]
    rvs = np.array(all_rvs)[idx]
    sigmas = np.array(all_sigmas)[idx]
    insts = [all_insts[i] for i in idx]

    return times, rvs, sigmas, insts


def load_vizier_data(sys_info: dict) -> tuple:
    """Load data from VizieR TSV files."""
    bjd_offset = sys_info.get("bjd_offset", 0.0)

    if "data_files" in sys_info:
        # Multi-file (e.g., HD 215152 pre/post upgrade)
        multi_insts = sys_info["multi_file_instruments"]
        all_times, all_rvs, all_sigmas, all_insts = [], [], [], []
        for fname, inst_label in zip(sys_info["data_files"], multi_insts):
            fpath = DOWNLOADS / "vizier" / fname
            t, rv, sig, _ = parse_vizier_tsv(
                fpath,
                rv_unit=sys_info.get("rv_unit", "km/s"),
                has_instr_col=False,
                default_instrument=inst_label,
                bjd_offset=bjd_offset,
            )
            all_times.extend(t.tolist())
            all_rvs.extend(rv.tolist())
            all_sigmas.extend(sig.tolist())
            all_insts.extend([inst_label] * len(t))

        idx = np.argsort(all_times)
        return (np.array(all_times)[idx], np.array(all_rvs)[idx],
                np.array(all_sigmas)[idx], [all_insts[i] for i in idx])
    else:
        fpath = DOWNLOADS / "vizier" / sys_info["data_file"]
        return parse_vizier_tsv(
            fpath,
            rv_unit=sys_info.get("rv_unit", "km/s"),
            has_instr_col=sys_info.get("has_instr_col", False),
            instr_map=sys_info.get("instr_map"),
            default_instrument=sys_info.get("default_instrument", "UNKNOWN"),
            bjd_offset=bjd_offset,
            has_ref_col=sys_info.get("has_ref_col", False),
            ref_instrument_map=sys_info.get("ref_instrument_map"),
        )


# =====================================================================
# l_rad computation (from fix_real_world_tasks.py)
# =====================================================================

def compute_l_rad_from_epoch(t_ref, P_days, omega_rad, e, T_epoch_jd, epoch_type="T_peri"):
    if epoch_type == "T_peri":
        M0 = (TWO_PI * ((t_ref - T_epoch_jd) / P_days)) % TWO_PI
    elif epoch_type == "T_conj":
        f_conj = math.pi / 2.0 - omega_rad
        M_conj = f_conj - 2.0 * e * math.sin(f_conj)
        M0 = (TWO_PI * ((t_ref - T_epoch_jd) / P_days) + M_conj) % TWO_PI
    else:
        raise ValueError(f"Unknown epoch_type: {epoch_type}")
    return float((omega_rad + M0) % TWO_PI)


def fit_l_rad_from_data(times, rvs, sigmas, P_days, K_ms):
    t_ref = times[0]
    phases = TWO_PI * (times - t_ref) / P_days
    best_l, best_chi2 = 0.0, np.inf
    for l_trial in np.linspace(0, TWO_PI, 360, endpoint=False):
        model = K_ms * np.sin(phases + l_trial)
        chi2 = np.sum(((rvs - model) / sigmas) ** 2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_l = l_trial
    return float(best_l % TWO_PI)


def build_planet_params(planet_info: dict, t_ref: float,
                        times=None, rvs=None, sigmas=None) -> dict:
    omega_rad = math.radians(planet_info["omega_deg"])

    if "T_peri_jd" in planet_info:
        l_rad = compute_l_rad_from_epoch(
            t_ref, planet_info["P_days"], omega_rad, planet_info["e"],
            planet_info["T_peri_jd"], epoch_type="T_peri"
        )
    elif "T_conj_jd" in planet_info:
        l_rad = compute_l_rad_from_epoch(
            t_ref, planet_info["P_days"], omega_rad, planet_info["e"],
            planet_info["T_conj_jd"], epoch_type="T_conj"
        )
    elif times is not None and rvs is not None and sigmas is not None:
        l_rad = fit_l_rad_from_data(times, rvs, sigmas,
                                     planet_info["P_days"], planet_info["K_ms"])
    else:
        l_rad = 0.0

    return {
        "P_days": planet_info["P_days"],
        "m_sin_i_mjup": planet_info["m_sin_i_mjup"],
        "e": planet_info["e"],
        "inc_rad": math.pi / 2.0,
        "Omega_rad": 0.0,
        "omega_rad": float(omega_rad),
        "l_rad": l_rad,
        "m_true_mjup": None,
    }


# =====================================================================
# Task generation
# =====================================================================

def build_task(task_id: str, sys_info: dict) -> Task:
    """Build a complete Task from system info and downloaded data."""
    print(f"\n{'='*60}")
    print(f"Building: {task_id}  ({sys_info['target_name']}, diff={sys_info['difficulty']})")

    # Load data
    if sys_info["data_source"] == "nasa_exoarchive":
        times, rvs, sigmas, instruments = load_nasa_data(sys_info)
    else:
        times, rvs, sigmas, instruments = load_vizier_data(sys_info)

    if len(times) == 0:
        raise ValueError(f"No data loaded for {task_id}")

    print(f"  Loaded {len(times)} observations")
    print(f"  Time range: JD {times[0]:.2f} to {times[-1]:.2f} ({times[-1]-times[0]:.1f} days)")
    print(f"  RV range: {rvs.min():.2f} to {rvs.max():.2f} m/s")
    print(f"  Median sigma: {np.median(sigmas):.2f} m/s")

    unique_insts = sorted(set(instruments))
    print(f"  Instruments: {unique_insts}")

    # Subtract systemic velocity (median per instrument, weighted)
    gamma_ms = float(np.median(rvs))
    rvs_centered = rvs - gamma_ms

    t_ref = float(times[0])

    # Build planet params
    planet_params_list = []
    for i, pinfo in enumerate(sys_info["planets"]):
        pp = build_planet_params(pinfo, t_ref, times, rvs_centered, sigmas)
        planet_params_list.append(pp)
        print(f"  Planet {i+1}: P={pp['P_days']:.4f}d, "
              f"m={pp['m_sin_i_mjup']:.4f}Mj, e={pp['e']:.3f}, "
              f"l_rad={pp['l_rad']:.4f}")

    # Build instrument params
    inst_params = []
    for inst_label in unique_insts:
        mask = np.array([i == inst_label for i in instruments])
        inst_sigma = float(np.median(sigmas[mask]))
        inst_gamma = float(np.median(rvs[mask]) - gamma_ms)
        inst_params.append({
            "label": inst_label,
            "gamma_ms": inst_gamma,
            "sigma_white_ms": inst_sigma,
            "sigma_jitter_ms": 0.0,
        })

    # Build the task dict (matching existing JSON structure exactly)
    task_dict = {
        "task_id": task_id,
        "truth_difficulty": sys_info["difficulty"],
        "difficulty_details": {
            "real_data": True,
            "n_planets": len(sys_info["planets"]),
            "n_instruments": len(unique_insts),
            "n_obs": len(times),
            "K_ms": max(p["K_ms"] for p in sys_info["planets"]),
            "challenge": sys_info.get("challenge", ""),
        },
        "config": {
            "engine": "keplerian",
            "star": {
                "M_star_sun": sys_info["M_star_sun"],
                "gamma_ms": gamma_ms,
            },
            "planets": planet_params_list,
            "schedule": {
                "times_days": times.tolist(),
                "instruments": instruments,
            },
            "instruments": inst_params,
            "noise": {
                "sigma_white_ms": float(np.median(sigmas)),
                "sigma_jitter_ms": 0.0,
                "gp": {
                    "use_gp": False,
                    "sigma_ms": 0.0,
                    "period_days": 25.0,
                    "Q0": 1.0,
                    "dQ": 0.1,
                    "f": 0.5,
                },
            },
            "los_axis": "x",
            "integrator_preference": "whfast",
            "metadata": {},
        },
        "observations": {
            "times_days": times.tolist(),
            "rvs_ms": rvs_centered.tolist(),
            "sigmas_ms": sigmas.tolist(),
            "instruments": instruments,
        },
        "meta": {
            "source": "real_observations",
            "rv_semantics": "rv_only",
            "gamma_subtracted_ms": gamma_ms,
            "n_obs": len(times),
            "baseline_days": float(times[-1] - times[0]),
            "target_name": sys_info["target_name"],
            "spectral_type": sys_info.get("spectral_type", ""),
        },
    }

    return task_dict


def main():
    print("=" * 60)
    print("Batch Generating Real-Data Task Bank JSON Files")
    print("=" * 60)

    BANK_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    errors = []

    for task_id, sys_info in SYSTEMS.items():
        try:
            task_dict = build_task(task_id, sys_info)
            out_path = BANK_DIR / f"{task_id}.json"
            with open(out_path, "w") as f:
                json.dump(task_dict, f, indent=2, sort_keys=True)
            print(f"  -> Saved: {out_path}")
            generated.append(task_id)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((task_id, str(e)))

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Generated: {len(generated)}/{len(SYSTEMS)}")
    for tid in generated:
        print(f"  ✅ {tid}")
    if errors:
        print(f"\nErrors: {len(errors)}")
        for tid, err in errors:
            print(f"  ❌ {tid}: {err}")

    # List all real-data tasks in bank
    print(f"\nAll real-data tasks in {BANK_DIR}:")
    for f in sorted(BANK_DIR.glob("*.json")):
        if f.name != "synthetic":
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
