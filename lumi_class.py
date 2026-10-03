
# %%
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  scipy.interpolate import interp1d
import json
import itertools
import numpy as np                          
import pandas as pd
import matplotlib.pyplot as plt
#https://github.com/PyCOMPLETE/FillingPatterns       
import fillingpatterns as fp   
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.dates as mdates
from scipy.constants import c as clight
from scipy.optimize import least_squares
import lumi_integrals
from importlib import reload
from datetime import datetime
import yaml
#from scipy.integrate import cumtrapz
try:
    from scipy.integrate import cumtrapz
except ImportError:
    # scipy >= 1.14 renamed cumtrapz to cumulative_trapezoid
    from scipy.integrate import cumulative_trapezoid as cumtrapz


params = {'xtick.labelsize': 22,
'ytick.labelsize': 22,
'font.size': 20,
          
'figure.autolayout': True,
'figure.figsize': (10, 8),
'axes.titlesize' : 30,
'axes.labelsize' : 30,
'lines.linewidth' : 4,
'lines.markersize' : 12,
'legend.fontsize': 15,
'mathtext.fontset': 'stix',
'font.family': 'STIXGeneral'}
plt.rcParams.update(params)

class LumiModel:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # general parameters
        self.verbose            = config["verbose"]
        self.frev               = 11245.5
        self.En                 = config["energy_GeV"]
        self.E0                 = 0.93828
        self.gamma              = self.En/self.E0
        self.betar              = 1
        self.VRF_V              = config["VRF_MV"]*1e6
        self.V0                 = self.VRF_V/1e9
        self.etap               = 0.0003481167216
        self.circ               = 26658.8832
        self.h                  = 35640
        self.nc                 = 1 # charge
        self.U0                 = 5.945310143e-06 # energy loss per turn


        # bunch length
        self.level_bl_at        = config["bunch_length"]["level_bl_at_s"]
        self.level_bl_to        = config["bunch_length"]["level_bl_to_s"]

        # IBS SR
        self.load_lookup_table  = config["IBS_SR"]["load_lookup_table"]
        self.lookup_table_path  = config["IBS_SR"]["lookup_table_path"]
        self.flag_lookup_xibs   = config["IBS_SR"]["flag_lookup_from_XIBS"]
        self.flag_SR_collider   = config["IBS_SR"]["flag_SR_collider"]
        self.collider_path      = config["IBS_SR"]["collider_json"]
        self.flag_IBS_xibs      = config["IBS_SR"]["flag_IBS_collider_with_xibs"]

        if config["IBS_SR"]["tauSRxy_hours"] is not None:
            self.tauSRxy_s          = config["IBS_SR"]["tauSRxy_hours"]*3600.0
        else:
            self.tauSRxy_s          = None
        if config["IBS_SR"]["tauSRl_hours"] is not None:
            self.tauSRl_s           = config["IBS_SR"]["tauSRl_hours"]*3600.0
        else:
            self.tauSRl_s           = None

        if (self.tauSRxy_s is None) or (self.tauSRl_s is None):

            if np.isclose(self.En, 6500.0):                        
                self.tauSRxy_s      = 64.7*3600;      
                self.tauSRl_s       = 32.35*3600;    
            elif np.isclose(self.En, 6800.0):
                self.tauSRxy_s      = 56.5*3600;      
                self.tauSRl_s       = 28.25*3600;    
            elif np.isclose(self.En, 7000.0):    
                self.tauSRxy_s      = 51.8*3600;     
                self.tauSRl_s       = 25.9*3600;  
            elif  np.isclose(self.En, 450.0):    
                self.tauSRxy_s      = 194970.7772*3600.0 
                self.tauSRl_s       = 97485.38859*3600.0 

        if self.flag_SR_collider: # computes the SR growth witn xtrack for the collider
            self.SR_collider()

        # extra emit growth
        self.ex_extra           = config["extra_emit_growth"]["ex_extra"]
        self.ey_extra           = config["extra_emit_growth"]["ey_extra"]
        self.use_emittance_data_x = config["extra_emit_growth"]["use_emittance_data_x"]
        self.use_emittance_data_y = config["extra_emit_growth"]["use_emittance_data_y"]

        # ATLAS and CMS
        self.year               = config["ATLAS_CMS"]["optics_year"]
        self.betaratio          = config["ATLAS_CMS"]["betaratio"]
        self.flag_xing_ct       = config["ATLAS_CMS"]["flag_xing_ct"]
        self.xing_ref           = config["ATLAS_CMS"]["xing_ref"]
        self.ATLAS_crossing     = config["ATLAS_CMS"]["ATLAS_crossing"]
        self.CMS_crossing       = config["ATLAS_CMS"]["CMS_crossing"]
        self.flag_lumi_simple   = config["ATLAS_CMS"]["flag_lumi_simple"]
        self.level_lumi_ip1     = float(config["ATLAS_CMS"]["level_lumi_ip1"])
        self.level_lumi_ip5     = float(config["ATLAS_CMS"]["level_lumi_ip5"])
        self.max_pileup_ip1     = config["ATLAS_CMS"]["max_pileup_ip1"]
        self.max_pileup_ip5     = config["ATLAS_CMS"]["max_pileup_ip5"]
        self.lumi_tol_percentage= config["ATLAS_CMS"]["lumi_tol_percentage"]
        self.level_virtual_lumi = config["ATLAS_CMS"]["level_virtual_lumi"]
        self.level_virtual_pu = config["ATLAS_CMS"]["level_virtual_pu"]
        self.virtual_lumi_emit  = float(config["ATLAS_CMS"]["virtual_lumi_emit"])
        self.virtual_lumi_tol_percentage = config["ATLAS_CMS"]["virtual_lumi_tol_percentage"]
        self.level_on_ATLAS_or_CMS = config["ATLAS_CMS"]["level_on_ATLAS_or_CMS"]

        self.beta_leveling_ended  = False
        self.separation_leveling_ended_ip1  = False
        self.separation_leveling_ended_ip5  = False
        self.xing_leveling_ended_ip1  = False
        self.xing_leveling_ended_ip5  = False
        self.beta_continuous_leveling_ended = False

        self.beta_min = 0
        if config["ATLAS_CMS"]["beta_min"] is not None:
            self.beta_min = config["ATLAS_CMS"]["beta_min"]

        # LHCb
        self.include_lhcb       = config["LHCb"]["include_lhcb"]
        self.xing_ip8_x         = config["LHCb"]["xing_ip8_x"]*2.0/1e6
        self.xing_ip8_y         = config["LHCb"]["xing_ip8_y"]*2.0/1e6
        self.betax_ip8          = config["LHCb"]["betax_ip8"]
        self.betay_ip8          = config["LHCb"]["betay_ip8"]
        self.level_lumi_ip8     = float(config["LHCb"]["level_lumi_ip8"])
        self.max_pileup_ip8     = float(config["LHCb"]["max_pileup_ip8"])
        self.lumi_tol_percentage_ip8 = config["LHCb"]["lumi_tol_percentage_ip8"]

        self.xing_ip8_r         = np.sqrt(self.xing_ip8_x**2 + self.xing_ip8_y**2)    
        self.phi_ip8            = np.arctan2(self.xing_ip8_y, self.xing_ip8_x)

        # cross sections
        self.sigma_el_m2        = config["cross_sections"]["sigma_el_m2"]
        self.sigmaBOff_m2       = config["cross_sections"]["sigmaBOff_m2"]
        self.sigmaBOff_losses   = config["cross_sections"]["sigmaBOff_losses"]

        # filling scheme
        self.compute_filling_scheme = config["filling_scheme"]["compute_filling_scheme"]
        self.number_of_bunches  = config["filling_scheme"]["number_of_bunches"]
        self.number_of_bunches_coll_ip15 = config["filling_scheme"]["number_of_bunches_coll_ip15"]
        self.number_of_bunches_coll_ip8 = config["filling_scheme"]["number_of_bunches_coll_ip8"]
        self.number_of_bunches_noncoll_ip15_b1 = config["filling_scheme"]["number_of_bunches_noncoll_ip15_b1"]
        self.number_of_bunches_noncoll_ip15_b2 = config["filling_scheme"]["number_of_bunches_noncoll_ip15_b2"]
        self.number_of_bunches_noncoll_ip8_b1 = config["filling_scheme"]["number_of_bunches_noncoll_ip8_b1"]
        self.number_of_bunches_noncoll_ip8_b2 = config["filling_scheme"]["number_of_bunches_noncoll_ip8_b2"]
        self.idx_b1             = config["filling_scheme"]["idx_b1"]
        self.idx_b2             = config["filling_scheme"]["idx_b2"]
        self.idx_b1_coll_ip15   = config["filling_scheme"]["idx_b1_coll_ip15"]
        self.idx_b2_coll_ip15   = config["filling_scheme"]["idx_b2_coll_ip15"]
        self.idx_b1_noncoll_ip15= config["filling_scheme"]["idx_b1_noncoll_ip15"]
        self.idx_b2_noncoll_ip15= config["filling_scheme"]["idx_b2_noncoll_ip15"]
        self.idx_b1_coll_ip8    = config["filling_scheme"]["idx_b1_coll_ip8"]
        self.idx_b2_coll_ip8    = config["filling_scheme"]["idx_b2_coll_ip8"]
        self.idx_b1_noncoll_ip8 = config["filling_scheme"]["idx_b1_noncoll_ip8"]
        self.idx_b2_noncoll_ip8 = config["filling_scheme"]["idx_b2_noncoll_ip8"]

        # coupling
        self.Coupling_coeff     = config["coupling"]["coupling_coeff"]
        self.Delta_tunesplit    = config["coupling"]["delta_tunesplit"]

        # crab cavities
        self.include_cc         = config["crab_cavities"]["include_cc"]
        if self.include_cc:
            self.ccx_ip1            = config["crab_cavities"]["ccx_ip1"]
            self.ccy_ip1            = config["crab_cavities"]["ccy_ip1"]
            self.ccx_ip5            = config["crab_cavities"]["ccx_ip5"]
            self.ccy_ip5            = config["crab_cavities"]["ccy_ip5"]
            self.ex_extra_CC        = config["crab_cavities"]["ex_extra_CC"]
            self.ey_extra_CC        = config["crab_cavities"]["ey_extra_CC"]
            self.VCC_MV             = config["crab_cavities"]["VCC_MV"]
        else:
            self.ccx_ip1, self.ccy_ip1, self.ccx_ip5, self.ccy_ip5, self.VCC_MV, self.ex_extra_CC, self.ey_extra_CC = 0, 0, 0, 0, 0, 0, 0

        # fraction of the crab angle applied to each beam (1 = full crabbing, 0 = no crabbing)
        self.cc_b1_fraction     = float(config["crab_cavities"].get("cc_b1_fraction", 1.0))
        self.cc_b2_fraction     = float(config["crab_cavities"].get("cc_b2_fraction", 1.0))

        if all([self.ccx_ip1 == 0, self.ccy_ip1 == 0, self.ccx_ip5 == 0, self.ccy_ip5 == 0]):
            self.include_cc = False

        # simulation parameters
        self.init_normalized_emittance = float(config["simulation_parameters"]["normalized_emittance"])
        self.init_intensity           = float(config["simulation_parameters"]["intensity"])
        self.init_bunch_length_s      = float(config["simulation_parameters"]["bunch_length_s"])
        self.time_step_minutes   = int(config["simulation_parameters"]["time_step_minutes"])
        self.time_periods        = int(config["simulation_parameters"]["time_periods"])

        # performance
        self.turnaround_time_h  = config["performance"]["turnaround_time_h"]
        self.efficiency         = config["performance"]["efficiency"]
        self.op_days            = config["performance"]["op_days"]


        # Initialize some values
        # Coupling
        self.Cfactor, self.dexy1, self.dexy2 = 0 , 0 , 0
        # Elastic scattering model
        self.dexdt1, self.deconvdt1, self.dexdt2, self.deconvdt2 = 0, 0 , 0 , 0

        self.on_sep1_x, self.on_sep1_y = 0, 0
        self.on_sep5_x, self.on_sep5_y = 0, 0
        self.on_sep8_x, self.on_sep8_y = 0, 0
        self.on_sep8_r = 0

        self.current_lumi_ATLAS, self.current_lumi_CMS, self.current_lumi_avg = 0, 0, 0
        self.current_pu_ATLAS, self.current_pu_CMS, self.current_pu_avg = 0, 0, 0

        self.current_lumi_LHCB = 0
        self.current_pu_LHCB = 0

        self.initial_beta_guess = 0.3

        if self.load_lookup_table:
            self.func_load_lookup_table()
            #self.BSRT_correction()
        
    ########################################################################################################
    ########################################################################################################

    def reset_values(self, **kwargs):
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Coupling emit BU model
        self.Cfactor, self.dexy1, self.dexy2 = 0 , 0 , 0
        # Elastic scattering model
        self.dexdt1, self.deconvdt1, self.dexdt2, self.deconvdt2 = 0, 0 , 0 , 0

        self.current_time = 0

        self.current_lumi_ATLAS, self.current_lumi_CMS, self.current_lumi_avg = 0, 0, 0
        self.current_pu_ATLAS, self.current_pu_CMS, self.current_pu_avg = 0, 0, 0

        self.current_lumi_LHCB = 0
        self.current_pu_LHCB = 0

        self.beta_leveling_ended  = False
        self.separation_leveling_ended_ip1  = False
        self.separation_leveling_ended_ip5  = False
        self.xing_leveling_ended_ip1  = False
        self.xing_leveling_ended_ip5  = False
        self.beta_continuous_leveling_ended = False

        self.on_sep1_x, self.on_sep1_y = 0, 0
        self.on_sep5_x, self.on_sep5_y = 0, 0
        self.on_sep8_x, self.on_sep8_y = 0, 0
        self.on_sep8_r = 0

        self.current_lumi_LHCB = 0
        self.current_pu_LHCB = 0

        self.initial_beta_guess = 0.3

        #if self.load_lookup_table:
        #    self.func_load_lookup_table()
        #    self.BSRT_correction()
               

    ########################################################################################################
    ########################################################################################################
    # BSRT correction

    def BSRT_correction(self):

        self.dict_case = {}
        self.dict_case['case'] = [self.BSRT_case]

        if self.dict_case['case'] == [1]: #uncorrected
            self.dict_case['cor_fact_1h'] = [1.]     
            self.dict_case['cor_fact_1v'] = [1.]
            self.dict_case['cor_fact_2h'] = [1.]
            self.dict_case['cor_fact_2v'] = [1.]
            print("case1: uncorrected BSRT")

        elif self.dict_case['case'] == [2]: #corrected
            #use the factors you want

            c1h = 0.9;c1v = 0.9;c2h = 0.9;c2v = 0.9; 

            self.dict_case['cor_fact_1h'] = [c1h]     
            self.dict_case['cor_fact_1v'] = [c1v]
            self.dict_case['cor_fact_2h'] = [c2h]
            self.dict_case['cor_fact_2v'] = [c2v]
            print("case2: correct BSRT by multiplying the emittances by a cor. factor")


        self.cor_fact_1h = self.dict_case['cor_fact_1h'][0]
        self.cor_fact_1v = self.dict_case['cor_fact_1v'][0]
        self.cor_fact_2h = self.dict_case['cor_fact_2h'][0]
        self.cor_fact_2v = self.dict_case['cor_fact_2v'][0]
        
        if self.verbose:
            print("HB1","","VB1","","HB2","","VB2")
            print(self.cor_fact_1h,self.cor_fact_1v,self.cor_fact_2h,self.cor_fact_2v)
            print()

        self.case = self.dict_case['case'][0]
    ########################################################################################################
    ########################################################################################################

    ########################################################################################################
    ########################################################################################################
    # Optics functions
    def cmp_xing_BBLR(self, x):
        a = 107.25867233
        b = 14.78092728
        c = 255.84493941
        d = -102.4652551
        return 0.5*(a + b*np.sqrt(x) + c*x + d*x**(3/2.))/(np.sqrt(x))

    def optics(self):

        if self.year == "2024":
            self.df_optics = pd.DataFrame({"betax":[1.200,1.125,1.055,0.990, 0.930,
                      0.875, 0.825, 0.775, 0.725, 
                      0.680, 0.640, 0.600, 0.560,
                      0.520,0.485,0.450,0.415,
                      0.385,0.355,0.325,0.300]})      
            self.df_optics["betay"] = self.df_optics["betax"]
        elif self.year == "2023":
            self.df_optics = pd.DataFrame({"betax":[1.200,1.125,1.055,0.990, 0.930,
                      0.875, 0.825, 0.775, 0.725, 
                      0.680, 0.640, 0.600, 0.560,
                      0.520,0.485,0.450,0.415,
                      0.385,0.355,0.325,0.300]})
            self.df_optics["betay"] = self.df_optics["betax"]
            
        elif self.year == "2022":
            self.df_optics = pd.DataFrame({"betax":[0.600, 0.560,
                      0.520,0.485,0.450,0.415,
                      0.385,0.355,0.325,0.300]})
            self.df_optics["betay"] = self.df_optics["betax"]
        
        elif self.year=="HLLHC":
             self.df_optics = pd.DataFrame({"betax":[0.15]})    
             if (self.beta_min < 0.15) and (self.beta_min > 0.0):
                self.df_optics.loc[len(self.df_optics)]  = {"betax": self.beta_min}
             self.df_optics["betay"] = self.df_optics["betax"]

        elif self.year == "2025":
            self.df_optics = pd.DataFrame({"betax": [1.2  , 1.125, 1.055, 0.99 , 0.93 , 0.875, 0.825, 0.775, 0.725,
       0.68 , 0.64 , 0.6  , 0.6  , 0.6  , 0.6  , 0.6  , 0.6  , 0.6  ,
       0.6  , 0.6  , 0.6  , 0.6  , 0.6  , 0.6  ] , "betay":  [1.2  , 1.125, 1.055, 0.99 , 0.93 , 0.875, 0.825, 0.775, 0.725,
       0.68 , 0.64 , 0.6  , 0.546, 0.492, 0.444, 0.402, 0.366, 0.33 ,
       0.3  , 0.27 , 0.246, 0.222, 0.198, 0.18 ] })

        if not self.flag_xing_ct:
            self.df_optics["xing"] = [self.cmp_xing_BBLR(i) for i in self.df_optics["betax"]]
        elif self.year == "2024":
            self.df_optics["xing"] = [150.0] + [160.0]*19 + [150.0]*1
        else:
            self.df_optics["xing"] = self.xing_ref      

        self.df_optics = self.df_optics[ (self.df_optics["betax"]>=self.beta_min) &
                                         (self.df_optics["betay"]>=self.beta_min) ]
        
        if len(self.df_optics) == 0:
            self.df_optics["betax"] = [self.beta_min]
            self.df_optics["betay"] = [self.beta_min]
        



            
        
    ########################################################################################################
    ########################################################################################################

    def sanity_checks(self):
        # to be filled and used
        assert not (self.betaratio != 1) or self.flag_xing_ct, "If betaratio is not 1, flag_xing_ct must be True."
        assert self.ATLAS_crossing != self.CMS_crossing, "ATLAS and CMS crossing angles must not be the same."


    ########################################################################################################
    ########################################################################################################
    # Initialise the Lumi model   
    def initialize(self, df_interpolated, bbb_or_avg = "avg", **kwargs):
        #self.__init__(load_lookup_table=False, **kwargs)
        self.reset_values(**kwargs)
        if "deactivate_optics" in kwargs:
            print("Not recomputing optics")
        else:
            self.optics()
                
        #self.optics()
        self.current_lookup_table = self.lookup_table[list(self.lookup_table.keys())[0]]

        if self.compute_filling_scheme and self.En>=6500:
            self.filling_scheme(df_interpolated)
        elif self.En>=6500.:

            if self.idx_b1 is None:
                # consider that all of them are colliding in IP15
                self.idx_b1 = np.where(df_interpolated['LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY'].iloc[0] > 0.)[0]
            if self.idx_b2 is None:
                self.idx_b2 = np.where(df_interpolated['LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY'].iloc[0] > 0.)[0]
            if self.number_of_bunches is None:
                self.number_of_bunches = len(self.idx_b1)
            if self.number_of_bunches_coll_ip15 is None:
                self.idx_b1_coll_ip15 = self.idx_b1
                self.idx_b2_coll_ip15 = self.idx_b2
                self.number_of_bunches_coll_ip15 = self.number_of_bunches
            #Considering that none of them are colliding in IP8 unless specified
            if (self.number_of_bunches_coll_ip8 is None) or (self.include_lhcb is False):
                self.idx_b1_coll_ip8 = np.array([])
                self.idx_b2_coll_ip8 = np.array([])
                self.number_of_bunches_coll_ip8 = 0

            if self.idx_b1_noncoll_ip15 is None:
                self.idx_b1_noncoll_ip15 = np.array([])
                self.number_of_bunches_noncoll_ip15_b1 = 0
            if self.idx_b2_noncoll_ip15 is None:
                self.idx_b2_noncoll_ip15 = np.array([])
                self.number_of_bunches_noncoll_ip15_b2 = 0
            if self.idx_b1_noncoll_ip8 is None:
                self.idx_b1_noncoll_ip8 = np.array([])
                self.number_of_bunches_noncoll_ip8_b1 = 0
            if self.idx_b2_noncoll_ip8 is None:
                self.idx_b2_noncoll_ip8 = np.array([])
                self.number_of_bunches_noncoll_ip8_b2 = 0
            
            if self.verbose:
                print()
                print("Filling scheme computation de-activated. Considering all bunches colliding in IP15 and none in IP8.")
                print(f"Total number of bunches {self.number_of_bunches}")
                print(f"Number of colliding bunches in IP15 {self.number_of_bunches_coll_ip15}")
                print(f"Number of non-colliding bunches in IP15 {self.number_of_bunches_noncoll_ip15_b1} and {self.number_of_bunches_noncoll_ip15_b2}")
                print()
                print("Colliding bunches in IP15 B1 and B2:", len(self.idx_b1_coll_ip15), len(self.idx_b2_coll_ip15))
                print("Non colliding bunches in IP15 B1 and B2:",  len(self.idx_b1_noncoll_ip15), len(self.idx_b2_noncoll_ip15))
                print()

                print(f"Number of colliding bunches in IP8 {self.number_of_bunches_coll_ip8}")
                print(f"Number of non-colliding bunches in IP8 {self.number_of_bunches_noncoll_ip8_b1} and {self.number_of_bunches_noncoll_ip8_b2}")
                print("Colliding bunches in IP8 B1 and B2:", len(self.idx_b1_coll_ip8), len(self.idx_b2_coll_ip8))
                print("Non colliding bunches in IP8 B1 and B2:",  len(self.idx_b1_noncoll_ip8), len(self.idx_b2_noncoll_ip8))  

        else:
            # Injection, get last point when machine is full
            if self.idx_b1 is None:
                self.idx_b1 = np.where(df_interpolated['LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY'].iloc[-1] > 0.)[0]
            if self.idx_b2 is None:
                self.idx_b2 = np.where(df_interpolated['LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY'].iloc[-1] > 0.)[0]

            # All bunches are non-colliding as we are at injection
            if self.idx_b1_noncoll_ip15 is None:
                self.idx_b1_noncoll_ip15 = self.idx_b1
            if self.idx_b2_noncoll_ip15 is None:
                self.idx_b1_noncoll_ip15 = self.idx_b2
            if self.idx_b1_noncoll_ip8 is None:
                self.idx_b1_noncoll_ip8 = np.array([])
            if self.idx_b2_noncoll_ip8 is None:
                self.idx_b2_noncoll_ip8 = np.array([])
            
            if self.number_of_bunches is None:
                self.number_of_bunches = len(self.idx_b1_noncoll_ip15)
                self.number_of_bunches_noncoll_ip15_b1 = self.number_of_bunches
                self.number_of_bunches_noncoll_ip15_b2 = self.number_of_bunches
                self.number_of_bunches_noncoll_ip8_b1 = 0
                self.number_of_bunches_noncoll_ip8_b2 = 0
            
            if self.verbose:
                print()
                print("Injection detected.")
                print(f"Number of bunches at the end of injection: {self.number_of_bunches}, {self.number_of_bunches_noncoll_ip15_b1}, {self.number_of_bunches_noncoll_ip15_b2}, {self.number_of_bunches_noncoll_ip8_b1}, {self.number_of_bunches_noncoll_ip8_b2}")
                print("Bunches in B1 and B2:",  len(self.idx_b1_noncoll_ip15), len(self.idx_b2_noncoll_ip15), len(self.idx_b1_noncoll_ip8), len(self.idx_b2_noncoll_ip8))   
                print()           
        
        if self.En>=6500:
            self.level_by_pu(how="ip1")
            self.level_by_pu(how="ip5") 

            if self.include_lhcb:
                self.level_by_pu(how="ip8")

        myvars = ['LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V', 
                'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_H',
                'LHC.BQM.B1:BUNCH_LENGTH_MEAN',
                'LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY']

        df_init = df_interpolated[myvars].copy()
        df_init.columns = ['ev_interp_coll', 'eh_interp_coll', 'bl_interp_m_coll', 'b_inten_interp_coll']
        df_init['bl_interp_m_coll'] = df_init['bl_interp_m_coll'].apply(lambda x: np.array([x/4.0*clight for i in range(3564)]) )
        
        df_init_old = df_init.copy()

        for myvar in ['ev_interp_coll', 'eh_interp_coll', 'bl_interp_m_coll', 'b_inten_interp_coll']:
            df_init[myvar] = df_init_old[myvar].apply(lambda x:x[self.idx_b1_coll_ip15])

        # to fixxxxx
        if self.number_of_bunches_noncoll_ip15_b1>0:
            for newvar, myvar in zip(['ev_interp_noncoll', 'eh_interp_noncoll', 'bl_interp_m_noncoll', 'b_inten_interp_noncoll'],
                                    ['ev_interp_coll', 'eh_interp_coll', 'bl_interp_m_coll', 'b_inten_interp_coll']):
                df_init[newvar] = df_init_old[myvar].apply(lambda x:x[self.idx_b1_noncoll_ip15])
        else:
            for newvar in ['ev_interp_noncoll', 'eh_interp_noncoll', 'bl_interp_m_noncoll', 'b_inten_interp_noncoll']:
                # equal to zero
                df_init[newvar] = None


        df_init['xing_angle_IP1'] = df_interpolated["LHC.RUNCONFIG:IP1-XING-V-MURAD"]*1e-6*2.0#xing_ip1*1e-6*2.0
        df_init['xing_angle_IP5'] = df_interpolated["LHC.RUNCONFIG:IP5-XING-H-MURAD"]*1e-6*2.0 #xing_ip5*1e-6*2.0
        df_init['betastar_IP1']   = df_interpolated['HX:BETASTAR_IP1'].values#beta_ip1
        df_init['betastar_IP5']   = df_interpolated['HX:BETASTAR_IP5'].values#beta_ip5
        df_init['beam'] = 'B1'
        #df_init = df_init.dropna()
        df_init_b1  = df_init.copy()

        myvars = ['LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_V', 
                'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_H',
                'LHC.BQM.B2:BUNCH_LENGTH_MEAN',
                'LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY']

        df_init = df_interpolated[myvars].copy()
        df_init.columns = ['ev_interp_coll', 'eh_interp_coll', 'bl_interp_m_coll', 'b_inten_interp_coll']
        df_init['bl_interp_m_coll'] = df_init['bl_interp_m_coll'].apply(lambda x: np.array([x/4.0*clight for i in range(3564)]) )

        df_init_old = df_init.copy()


        for myvar in ['ev_interp_coll', 'eh_interp_coll', 'bl_interp_m_coll', 'b_inten_interp_coll']:
            df_init[myvar] = df_init_old[myvar].apply(lambda x:x[self.idx_b2_coll_ip15])

        # to fixxxxx
        if self.number_of_bunches_noncoll_ip15_b2>0:
            for newvar, myvar in zip(['ev_interp_noncoll', 'eh_interp_noncoll', 'bl_interp_m_noncoll', 'b_inten_interp_noncoll'],
                                    ['ev_interp_coll', 'eh_interp_coll', 'bl_interp_m_coll', 'b_inten_interp_coll']):
                df_init[newvar] = df_init_old[myvar].apply(lambda x:x[self.idx_b2_noncoll_ip15])
        else:
            for newvar in ['ev_interp_noncoll', 'eh_interp_noncoll', 'bl_interp_m_noncoll', 'b_inten_interp_noncoll']:
                df_init[newvar] = None

        df_init['xing_angle_IP1'] = df_interpolated["LHC.RUNCONFIG:IP1-XING-V-MURAD"]*1e-6*2.0#xing_ip1*1e-6*2.0
        df_init['xing_angle_IP5'] = df_interpolated["LHC.RUNCONFIG:IP5-XING-H-MURAD"]*1e-6*2.0 #xing_ip5*1e-6*2.0
        df_init['betastar_IP1']   = df_interpolated['HX:BETASTAR_IP1'].values
        df_init['betastar_IP5']   = df_interpolated['HX:BETASTAR_IP5'].values
        df_init['beam'] = 'B2'
        #df_init = df_init.dropna()
        df_init_b2  = df_init.copy()

        # rescale emittances based on factor
        #for myvar in ['eh_interp_coll',  'eh_interp_noncoll']:
        #    if not df_init_b1[myvar].isnull().all():
        #        df_init_b1[myvar] = df_init_b1[myvar].apply(lambda x:x*self.cor_fact_1h)
        #    if not df_init_b2[myvar].isnull().all():
        #        df_init_b2[myvar] = df_init_b2[myvar].apply(lambda x:x*self.cor_fact_2h)
            
        #for myvar in ['ev_interp_coll',  'ev_interp_noncoll']:
        #    if not df_init_b1[myvar].isnull().all():
        #        df_init_b1[myvar] = df_init_b1[myvar].apply(lambda x:x*self.cor_fact_1v)
        #    if not df_init_b2[myvar].isnull().all():
        #        df_init_b2[myvar] = df_init_b2[myvar].apply(lambda x:x*self.cor_fact_2v)


        self.df_init = pd.concat([df_init_b1, df_init_b2])

        if bbb_or_avg == "avg":

            def get_avg(x, var):
                return np.array([np.mean(x[var])])
                
            for var in ["eh_interp_coll", "ev_interp_coll","bl_interp_m_coll",     "b_inten_interp_coll",
                        "eh_interp_noncoll", "ev_interp_noncoll", "bl_interp_m_noncoll", "b_inten_interp_noncoll"]:
                if not self.df_init[var].isnull().all():
                    self.df_init[var] = self.df_init.apply(get_avg, var=var, axis=1)
                

        self.dt_step = self.df_init.index.to_series().diff()[1].total_seconds()
        
        self.df_init_b1 = self.df_init[self.df_init['beam']=='B1']
        self.df_init_b2 = self.df_init[self.df_init['beam']=='B2']

        self.steps = len(self.df_init_b1)

        if self.verbose:
            print("")
            print(f"Time step {self.dt_step/60.} min")
            print(f"Initial conditions: \n For B1: \n {self.df_init_b1.iloc[0]}")
            print(f"For B2: \n {self.df_init_b2.iloc[0]}")
            print("")


        self.initialize_results()

    
    def initialize_results(self):
        #self.results_beta = np.empty(self.steps)
        self.results_betax = np.empty(self.steps)
        self.results_betay = np.empty(self.steps)
        
        self.results_betax_ip1 = np.empty(self.steps)
        self.results_betay_ip1 = np.empty(self.steps)
        self.results_betax_ip5 = np.empty(self.steps)
        self.results_betay_ip5 = np.empty(self.steps)
        
        self.results_xing = np.empty(self.steps)
        self.results_xing_ip1_x = np.empty(self.steps)
        self.results_xing_ip1_y = np.empty(self.steps)
        self.results_xing_ip5_x = np.empty(self.steps)
        self.results_xing_ip5_y = np.empty(self.steps)
        self.results_on_sep1_x = np.zeros(self.steps)
        self.results_on_sep1_y = np.zeros(self.steps)
        self.results_on_sep5_x = np.zeros(self.steps)
        self.results_on_sep5_y = np.zeros(self.steps)
        self.results_emitx_B1 = [[] for _ in range(self.steps)]
        self.results_emity_B1 = [[] for _ in range(self.steps)]
        self.results_emitx_B2 = [[] for _ in range(self.steps)]
        self.results_emity_B2 = [[] for _ in range(self.steps)]
        self.results_emitx_B1_uncoupled = [[] for _ in range(self.steps)]
        self.results_emity_B1_uncoupled = [[] for _ in range(self.steps)]
        self.results_emitx_B2_uncoupled = [[] for _ in range(self.steps)]
        self.results_emity_B2_uncoupled = [[] for _ in range(self.steps)]
        self.results_bl_s_B1 = [[] for _ in range(self.steps)]
        self.results_bl_s_B2 = [[] for _ in range(self.steps)]
        self.results_inten_B1 = [[] for _ in range(self.steps)]
        self.results_inten_B2 = [[] for _ in range(self.steps)]
        self.results_lumi_ATLAS = [[] for _ in range(self.steps)]
        self.results_lumi_CMS = [[] for _ in range(self.steps)]
        self.results_lumi_avg = np.empty(self.steps)
        self.results_pu_ATLAS = [[] for _ in range(self.steps)]
        self.results_pu_CMS = [[] for _ in range(self.steps)]
        self.results_pu_avg = np.empty(self.steps)

        self.results_lumi_ATLAS_virtual = [[] for _ in range(self.steps)]
        self.results_lumi_CMS_virtual = [[] for _ in range(self.steps)]
        self.results_lumi_avg_virtual = np.empty(self.steps)
        self.results_pu_ATLAS_virtual = [[] for _ in range(self.steps)]
        self.results_pu_CMS_virtual = [[] for _ in range(self.steps)]
        self.results_pu_avg_virtual = np.empty(self.steps)

        #self.results_longitudinal_luminous_region_ATLAS = [[] for _ in range(self.steps)]
        #self.results_longitudinal_luminous_region_CMS = [[] for _ in range(self.steps)]
        #self.results_longitudinal_luminous_region_avg = np.empty(self.steps)

        self.results_lumi_ATLAS_virtual_2um = [[] for _ in range(self.steps)]
        self.results_lumi_CMS_virtual_2um = [[] for _ in range(self.steps)]
        self.results_lumi_avg_virtual_2um = np.empty(self.steps)
        self.results_pu_ATLAS_virtual_2um = [[] for _ in range(self.steps)]
        self.results_pu_CMS_virtual_2um = [[] for _ in range(self.steps)]
        self.results_pu_avg_virtual_2um = np.empty(self.steps)

        self.results_lumi_LHCB = [[] for _ in range(self.steps)]
        self.results_pu_LHCB = [[] for _ in range(self.steps)]
        self.results_betax_ip8 = np.empty(self.steps)
        self.results_betay_ip8 = np.empty(self.steps)
        self.results_xing_ip8_x = np.empty(self.steps)
        self.results_xing_ip8_y = np.empty(self.steps)
        self.results_xing_ip8_r = np.empty(self.steps)
        self.results_on_sep8_x = np.zeros(self.steps)
        self.results_on_sep8_y = np.zeros(self.steps)
        self.results_on_sep8_r = np.zeros(self.steps)

        self.results_ccvx_ip1 = np.zeros(self.steps)
        self.results_ccvy_ip1 = np.zeros(self.steps)
        self.results_ccvx_ip5 = np.zeros(self.steps)
        self.results_ccvy_ip5 = np.zeros(self.steps)



    ########################################################################################################
    ########################################################################################################
    # Filling scheme

    def filling_scheme(self, df):
            #var_b1 = 'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_H'
            #var_b2 = 'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_H'
            var_b1 = "LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY"
            var_b2 = "LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY"

            mask = df[var_b1].dropna().iloc[0]
            idx_b1 = np.where(mask>0.)[0]
            mask = df[var_b2].dropna().iloc[0]
            idx_b2 = np.where(mask>0.)[0]

            filling_scheme = {}                                                                                  
            for beam in ['beam1','beam2']:
                filling_scheme[beam] = np.zeros(3564, dtype=int)

                if beam=='beam1':
                    filling_scheme[beam][idx_b1]=1
                else:
                    filling_scheme[beam][idx_b2]=1

                filling_scheme[beam] = filling_scheme[beam].tolist()         
                    
            self.number_of_bunches = len(idx_b1)
                
            patt = fp.FillingPattern(filling_scheme["beam1"], filling_scheme["beam2"])               
            patt.compute_beam_beam_schedule(n_lr_per_side=21) 
            self.patt = patt

            self.idx_b1 = idx_b1
            self.idx_b2 = idx_b2

            self.idx_b1_coll_ip15 = np.array(patt.b1.bb_schedule[patt.b1.bb_schedule['collides in ATLAS/CMS']==True]["HO partner in ATLAS/CMS"].index).astype(int)
            self.idx_b2_coll_ip15 = np.array(patt.b2.bb_schedule[patt.b2.bb_schedule['collides in ATLAS/CMS']==True]["HO partner in ATLAS/CMS"].index).astype(int)

            self.idx_b1_noncoll_ip15 = np.array(patt.b1.bb_schedule[patt.b1.bb_schedule['collides in ATLAS/CMS']==False]["HO partner in ATLAS/CMS"].index).astype(int)
            self.idx_b2_noncoll_ip15 = np.array(patt.b2.bb_schedule[patt.b2.bb_schedule['collides in ATLAS/CMS']==False]["HO partner in ATLAS/CMS"].index).astype(int)
                
            self.number_of_bunches_coll_ip15 = len(self.idx_b1_coll_ip15)
            self.number_of_bunches_noncoll_ip15 = len(self.idx_b1_noncoll_ip15)
                

            self.idx_b1_coll_ip8 = np.array(patt.b1.bb_schedule[patt.b1.bb_schedule['collides in LHCB']==True]["HO partner in LHCB"].index).astype(int)
            self.idx_b2_coll_ip8 = np.array(patt.b2.bb_schedule[patt.b2.bb_schedule['collides in LHCB']==True]["HO partner in LHCB"].index).astype(int)

            self.idx_b1_noncoll_ip8 = np.array(patt.b1.bb_schedule[patt.b1.bb_schedule['collides in LHCB']==False]["HO partner in LHCB"].index).astype(int)
            self.idx_b2_noncoll_ip8 = np.array(patt.b2.bb_schedule[patt.b2.bb_schedule['collides in LHCB']==False]["HO partner in LHCB"].index).astype(int)
                
            self.number_of_bunches_coll_ip8 = len(self.idx_b1_coll_ip8)
            self.number_of_bunches_noncoll_ip8 = len(self.idx_b1_noncoll_ip8)
                
            if self.verbose:
                print()
                print("Colliding bunches in IP15 for B1 and B2:", len(self.idx_b1_coll_ip15), len(self.idx_b2_coll_ip15))
                print("Non colliding bunches in IP15 for B1 and B2:",  len(self.idx_b1_noncoll_ip15), len(self.idx_b2_noncoll_ip15)) 
                print()
    ########################################################################################################
    ########################################################################################################
    
    def use_data(self,  bunch_family="coll", current_time_step=0, use_emittance_x = False, use_emittance_y=False, use_intensity = False, use_bl = False, use_beta=False):
        
        if use_emittance_x:
            self.current_emitx_B1 = self.df_init_b1[f"eh_interp_{bunch_family}"].iloc[current_time_step]*1e-6
            #self.current_emity_B1 = self.df_init_b1[f"ev_interp_{bunch_family}"].iloc[current_time_step]*1e-6
            self.current_emitx_B2 = self.df_init_b2[f"eh_interp_{bunch_family}"].iloc[current_time_step]*1e-6
            #self.current_emity_B2 = self.df_init_b2[f"ev_interp_{bunch_family}"].iloc[current_time_step]*1e-6 
            self.current_emitx_B1_uncoupled = self.current_emitx_B1
            #self.current_emity_B1_uncoupled = self.current_emity_B1
            self.current_emitx_B2_uncoupled = self.current_emitx_B2
            #self.current_emity_B2_uncoupled = self.current_emity_B2
        if use_emittance_y:
            self.current_emity_B1 = self.df_init_b1[f"ev_interp_{bunch_family}"].iloc[current_time_step]*1e-6
            self.current_emity_B2 = self.df_init_b2[f"ev_interp_{bunch_family}"].iloc[current_time_step]*1e-6
            self.current_emity_B1_uncoupled = self.current_emity_B1
            self.current_emity_B2_uncoupled = self.current_emity_B2
        if use_bl:
            self.current_bl_s_B1 = self.df_init_b1[f"bl_interp_m_{bunch_family}"].iloc[current_time_step]*4.0/clight
            self.current_bl_s_B2 = self.df_init_b2[f"bl_interp_m_{bunch_family}"].iloc[current_time_step]*4.0/clight
        if use_intensity:
            self.current_inten_B1 = self.df_init_b1[f"b_inten_interp_{bunch_family}"].iloc[current_time_step]
            self.current_inten_B2 = self.df_init_b2[f"b_inten_interp_{bunch_family}"].iloc[current_time_step]
        if use_beta:
            self.current_betax_ip1 = self.df_init_b1.iloc[current_time_step]["betastar_IP1"]*1e-2
            self.current_betay_ip1 = self.df_init_b1.iloc[current_time_step]["betastar_IP1"]*1e-2
            self.current_betax_ip5 = self.df_init_b1.iloc[current_time_step]["betastar_IP5"]*1e-2
            self.current_betay_ip5 = self.df_init_b1.iloc[current_time_step]["betastar_IP5"]*1e-2
            self.current_xing_ip1_y = self.df_init_b1.iloc[current_time_step]["xing_angle_IP1"]
            self.current_xing_ip5_x = self.df_init_b1.iloc[current_time_step]["xing_angle_IP5"]
    ########################################################################################################
    ########################################################################################################

    def initialize_t0(self, bunch_family="coll", current_time_step=0):
        self.current_betax = self.df_init_b1["betastar_IP1"].iloc[current_time_step]*1e-2
        self.current_betay = self.df_init_b1["betastar_IP1"].iloc[current_time_step]*1e-2
        self.current_betax_ip1 = self.df_init_b1["betastar_IP1"].iloc[current_time_step]*1e-2
        self.current_betay_ip1 = self.df_init_b1["betastar_IP1"].iloc[current_time_step]*1e-2
        self.current_betax_ip5 = self.df_init_b1["betastar_IP5"].iloc[current_time_step]*1e-2
        self.current_betay_ip5 = self.df_init_b1["betastar_IP5"].iloc[current_time_step]*1e-2

        self.current_xing = self.df_init_b1["xing_angle_IP1"].iloc[current_time_step]

        self.current_emitx_B1 = self.df_init_b1[f"eh_interp_{bunch_family}"].iloc[current_time_step]*1e-6
        self.current_emity_B1 = self.df_init_b1[f"ev_interp_{bunch_family}"].iloc[current_time_step]*1e-6
        self.current_emitx_B2 = self.df_init_b2[f"eh_interp_{bunch_family}"].iloc[current_time_step]*1e-6
        self.current_emity_B2 = self.df_init_b2[f"ev_interp_{bunch_family}"].iloc[current_time_step]*1e-6

        self.current_bl_s_B1 = self.df_init_b1[f"bl_interp_m_{bunch_family}"].iloc[current_time_step]*4.0/clight
        self.current_bl_s_B2 = self.df_init_b2[f"bl_interp_m_{bunch_family}"].iloc[current_time_step]*4.0/clight
        self.current_inten_B1 = self.df_init_b1[f"b_inten_interp_{bunch_family}"].iloc[current_time_step]
        self.current_inten_B2 = self.df_init_b2[f"b_inten_interp_{bunch_family}"].iloc[current_time_step]

        for attr in ["current_on_sep1_x", "current_on_sep1_y", "current_on_sep5_x", "current_on_sep5_y"]:
            setattr(self, attr, 0.0)
        self.beta_idx = 0

        self.current_emitx_B1_uncoupled = self.current_emitx_B1
        self.current_emity_B1_uncoupled = self.current_emity_B1
        self.current_emitx_B2_uncoupled = self.current_emitx_B2
        self.current_emity_B2_uncoupled = self.current_emity_B2

        if self.include_lhcb:
            self.current_betax_ip8 = self.betax_ip8
            self.current_betay_ip8 = self.betay_ip8
            self.current_xing_ip8_x = self.xing_ip8_x
            self.current_xing_ip8_y = self.xing_ip8_y
            self.current_xing_ip8_r = self.xing_ip8_r
            self.current_on_sep8_x = self.on_sep8_x
            self.current_on_sep8_y = self.on_sep8_y
            self.current_on_sep8_r = self.on_sep8_r

        # Adjust for ATLAS crossing
        if self.ATLAS_crossing == "V":
            self.current_betax_ip1, self.current_betay_ip1 = sorted([self.current_betax, self.current_betay])
            self.current_xing_ip1_y, self.current_xing_ip1_x = self.current_xing, 0.0
        else:
            self.current_betay_ip1, self.current_betax_ip1 = sorted([self.current_betax, self.current_betay])
            self.current_xing_ip1_x, self.current_xing_ip1_y = self.current_xing, 0.0

        # Adjust for CMS crossing
        if self.CMS_crossing == "H":
            self.current_betax_ip5, self.current_betay_ip5 = sorted([self.current_betax, self.current_betay], reverse=True)
            self.current_xing_ip5_x, self.current_xing_ip5_y = self.current_xing, 0.0
        else:
            self.current_betay_ip5, self.current_betax_ip5 = sorted([self.current_betax, self.current_betay], reverse=True)
            self.current_xing_ip5_y, self.current_xing_ip5_x = self.current_xing, 0.0

        if bunch_family=="coll":

            self.current_lumi_ATLAS, self.current_lumi_CMS = self.compute_CMS_ATLAS_lumi()
            self.current_pu_ATLAS = self.cmp_pileup(self.current_lumi_ATLAS)
            self.current_pu_CMS = self.cmp_pileup(self.current_lumi_CMS)

            for attr in ["current_lumi_CMS_virtual", "current_lumi_ATLAS_virtual", "current_pu_CMS_virtual", "current_pu_ATLAS_virtual",
                     "current_lumi_CMS_virtual_2um", "current_lumi_ATLAS_virtual_2um", "current_pu_CMS_virtual_2um", "current_pu_ATLAS_virtual_2um"]:
                setattr(self, attr, 0)

            if self.include_lhcb:
                self.current_lumi_LHCB  = self.Lumi_integrals(self.number_of_bunches_coll_ip8, 
                                           betax = self.current_betax_ip8, betay= self.current_betay_ip8, 
                                           x=self.current_on_sep8_x, px=self.current_xing_ip8_x/2., y=self.current_on_sep8_y, py=self.current_xing_ip8_y/2., verbose=False)

                self.current_pu_LHCB = self.cmp_pileup(self.current_lumi_LHCB, how="ip8")



    # Useful functions




    # Leveling by PU
    def level_by_pu(self, how="both"):
        #if how=="both":
        #    if self.max_pileup:
        #        level_lumi_from_pileup               = self.max_pileup*self.number_of_bunches_coll_ip15*self.frev/(self.sigmaBOff_m2)
        #        if not np.isclose((min(self.level_lumi, level_lumi_from_pileup)), self.level_lumi):
        #            print(f"Leveling IP15 lumi at {level_lumi_from_pileup} instead of {self.level_lumi} due to max pileup set at {self.max_pileup}")
        #        self.level_lumi = (min(self.level_lumi, level_lumi_from_pileup))
        #        print(f"Leveled IP15 lumi from {self.level_lumi-self.lumi_tol_percentage/100.*self.level_lumi} to {self.level_lumi+self.lumi_tol_percentage/100.*self.level_lumi}")
        if how=="ip1":
            if self.max_pileup_ip1:
                level_lumi_from_pileup_ip1           = self.max_pileup_ip1*self.number_of_bunches_coll_ip15*self.frev/(self.sigmaBOff_m2)
                if not np.isclose((min(self.level_lumi_ip1, level_lumi_from_pileup_ip1)), self.level_lumi_ip1):
                    print(f"Leveling IP1 lumi at {level_lumi_from_pileup_ip1} instead of {self.level_lumi_ip1} due to max pileup set at {self.max_pileup_ip1}")
                self.level_lumi_ip1 = (min(self.level_lumi_ip1, level_lumi_from_pileup_ip1))
                print(f"Leveled IP1 lumi from {self.level_lumi_ip1-self.lumi_tol_percentage/100.*self.level_lumi_ip1} to {self.level_lumi_ip1+self.lumi_tol_percentage/100.*self.level_lumi_ip1}")
        elif how=="ip5":
            if self.max_pileup_ip5:
                level_lumi_from_pileup_ip5           = self.max_pileup_ip5*self.number_of_bunches_coll_ip15*self.frev/(self.sigmaBOff_m2)
                if not np.isclose((min(self.level_lumi_ip5, level_lumi_from_pileup_ip5)), self.level_lumi_ip5):
                    print(f"Leveling IP5 lumi at {level_lumi_from_pileup_ip5} instead of {self.level_lumi_ip5} due to max pileup set at {self.max_pileup_ip5}")
                self.level_lumi_ip5 = (min(self.level_lumi_ip5, level_lumi_from_pileup_ip5))
                print(f"Leveled IP5 lumi from {self.level_lumi_ip5-self.lumi_tol_percentage/100.*self.level_lumi_ip5} to {self.level_lumi_ip5+self.lumi_tol_percentage/100.*self.level_lumi_ip5}")
        elif how=="ip8":
            if self.max_pileup_ip8:
                level_lumi_from_pileup_ip8           = self.max_pileup_ip8*self.number_of_bunches_coll_ip8*self.frev/(self.sigmaBOff_m2)
                if not np.isclose((min(self.level_lumi_ip8, level_lumi_from_pileup_ip8)), self.level_lumi_ip8):
                    print(f"Leveling IP8 lumi at {level_lumi_from_pileup_ip8} instead of {self.level_lumi_ip8} due to max pileup set at {self.max_pileup_ip8}")
                self.level_lumi_ip8 = (min(self.level_lumi_ip8, level_lumi_from_pileup_ip8))
                print(f"Leveled IP8 lumi from {self.level_lumi_ip8-self.lumi_tol_percentage/100.*self.level_lumi_ip8} to {self.level_lumi_ip8+self.lumi_tol_percentage/100.*self.level_lumi_ip8}")
        elif how=="virtual":
            if self.level_virtual_pu:
                level_lumi_from_pileup_virtual           = self.level_virtual_pu*self.number_of_bunches_coll_ip15*self.frev/(self.sigmaBOff_m2)
                if not np.isclose((min(self.level_virtual_lumi, level_lumi_from_pileup_virtual)), self.level_virtual_lumi):
                    print(f"Leveling virtual lumi at {level_lumi_from_pileup_virtual} instead of {self.level_virtual_lumi} due to max pileup set at {self.level_virtual_pu}")
                self.level_virtual_lumi = (min(self.level_virtual_lumi, level_lumi_from_pileup_virtual))
                print(f"Leveled virtual lumi from {self.level_virtual_lumi-self.lumi_tol_percentage/100.*self.level_virtual_lumi} to {self.level_virtual_lumi+self.lumi_tol_percentage/100.*self.level_virtual_lumi}")
        else:
            raise ValueError("how should be 'both', 'ip1' or 'ip5'")


    def print_current_values(self):
        print("")
        print("Current values:")
        print(f"Time step {self.current_time} min")
        #print("Current beta*:", self.current_beta)
        print("Current betax*:", self.current_betax)
        print("Current betay*:", self.current_betay)
        print("Current crossing angle:", self.current_xing)
        print("Current crossing angle IP1 H:", self.current_xing_ip1_x)
        print("Current crossing angle IP1 V:", self.current_xing_ip1_y)
        print("Current crossing angle IP5 H:", self.current_xing_ip5_x)
        print("Current crossing angle IP5 V:", self.current_xing_ip5_y)
        print("Current separation IP1:", self.current_on_sep1_x)
        print("Current separation IP5:", self.current_on_sep1_y)
        print("Current separation IP5:", self.current_on_sep5_x)
        print("Current separation IP5:", self.current_on_sep5_y)
        print("Current emitx B1:", self.current_emitx_B1)
        print("Current emity B1:", self.current_emity_B1)
        print("Current emitx B2:", self.current_emitx_B2)
        print("Current emity B2:", self.current_emity_B2)
        print("Current emitx B1 uncoupled:", self.current_emitx_B1_uncoupled)
        print("Current emity B1 uncoupled:", self.current_emity_B1_uncoupled)
        print("Current emitx B2 uncoupled:", self.current_emitx_B2_uncoupled)
        print("Current emity B2 uncoupled:", self.current_emity_B2_uncoupled)
        print("Current bl_s B1:", self.current_bl_s_B1)
        print("Current bl_s B2:", self.current_bl_s_B2)
        print("Current inten B1:", self.current_inten_B1)
        print("Current inten B2:", self.current_inten_B2)
        print("Current lumi ATLAS:", self.current_lumi_ATLAS)
        print("Current lumi CMS:", self.current_lumi_CMS)
        print("Current lumi avg:", np.mean([np.mean(self.current_lumi_ATLAS), np.mean(self.current_lumi_CMS)]))
        print("Current pu ATLAS:", self.current_pu_ATLAS)
        print("Current pu CMS:", self.current_pu_CMS)
        print("Current pu avg:", np.mean([np.mean(self.current_pu_ATLAS), np.mean(self.current_pu_CMS)])
)
        
        if self.include_lhcb:
            print("Current lumi LHCb:", self.current_lumi_LHCB)
            print("Current pu LHCb:", self.current_pu_LHCB)
            print("Current betax* IP8:", self.current_betax_ip8)
            print("Current betay* IP8:", self.current_betay_ip8)
            print("Current crossing angle IP8:", self.current_xing_ip8_x, self.current_xing_ip8_y, self.current_xing_ip8_r)
            print("Current separation IP8:", self.current_on_sep8_x, self.current_on_sep8_y, self.current_on_sep8_r)

        try:
            print("IBS growth rates B1: ", self.current_Tx_B1, self.current_Ty_B1, self.current_Tl_B1)
            print("IBS growth rates B2: ", self.current_Tx_B2, self.current_Ty_B2, self.current_Tl_B2)
            print("Elastic scatter: ", self.dexdt1, self.dexdt2)
            print("Coupling: ", self.dexy1, self.dexy2, self.Cfactor)
            print("Noise: ", self.dex_noise_B1, self.dex_noise_B2, self.dey_noise_B1, self.dey_noise_B2)
            print("BOFF: ", self.dex_noise_B1, self.dey_noise_B1, self.dex_noise_B2, self.dey_noise_B2)
         
        except:
            pass
        print("")

    def Lumi_inst_IP15(self,
                  bunch_intensity_p1, bunch_intensity_p2, 
                  ex_norm_m1, ey_norm_m1, ex_norm_m2, ey_norm_m2, 
                  bl_4sigma_s1, bl_4sigma_s2, 
                  betastar_m_x, betastar_m_y, 
                  phi_full_rad, experiment,
                  sepx = 0, sepy = 0, return_factors=False):
        
        sigmaz_m = (bl_4sigma_s1+bl_4sigma_s2)/2./4.*clight

        sigmax_m = np.sqrt(ex_norm_m1/self.gamma*betastar_m_x+ex_norm_m2/self.gamma*betastar_m_x)/np.sqrt(2.)
        sigmay_m = np.sqrt(ey_norm_m1/self.gamma*betastar_m_y+ey_norm_m2/self.gamma*betastar_m_y)/np.sqrt(2.)

        if not experiment:
            experiment = 'CMS'

        # HO

        if experiment=='ATLAS':
            if self.ATLAS_crossing == "V":
                FF=1./((1.+(sigmaz_m/sigmay_m*phi_full_rad/2.)**2)**0.5);
                W = np.exp(-sepx**2/sigmax_m**2)
            else:
                FF=1./((1.+(sigmaz_m/sigmax_m*phi_full_rad/2.)**2)**0.5);
                W = np.exp(-sepy**2/sigmay_m**2)
        elif experiment=='CMS':
            if self.CMS_crossing == "V":
                FF=1./((1.+(sigmaz_m/sigmay_m*phi_full_rad/2.)**2)**0.5);
                W = np.exp(-sepx**2/sigmax_m**2)
            else:
                FF=1./((1.+(sigmaz_m/sigmax_m*phi_full_rad/2.)**2)**0.5);
                W = np.exp(-sepy**2/sigmay_m**2)

        lumi_HO = self.frev*self.number_of_bunches_coll_ip15*bunch_intensity_p1*bunch_intensity_p2/(sigmax_m*sigmay_m)/4./np.pi

        if return_factors:
            return FF*lumi_HO*W, lumi_HO, FF, W
        else:
            return FF*lumi_HO*W
    

    def Lumi_integrals(self, number_of_bunches, betax = 0, betay= 0, x=0, px=0, y=0, py=0, ccx=0, ccy=0, ccrf=400e6,verbose=False, emit=None, flag="lumi"):
  
        ccr12 = 1
        ccr34 = 1
        ccvx = ccx / ccr12 * self.En*1e9 / (2.0*np.pi) / ccrf * clight
        ccvy = ccy / ccr34 * self.En*1e9 / (2.0*np.pi) / ccrf * clight

        self.ccvx = ccvx
        self.ccvy = ccvy

        sigma_z1 = self.current_bl_s_B1/4.*clight
        sigma_z2 = self.current_bl_s_B2/4.*clight

        if verbose:
            print("sigma_z1", sigma_z1)
            print("sigma_z2", sigma_z2)
            print("nb", number_of_bunches)
            print("N1", self.current_inten_B1)
            print("N2", self.current_inten_B2)
            print("x_1", x)
            print("x_2", -x)
            print("y_1", y)
            print("y_2", -y)
            print("px_1", px)
            print("px_2", -px)
            print("py_1", py)
            print("py_2", -py)
            print("energy_tot1", self.En*1e9)
            print("energy_tot2", self.En*1e9)
            print("deltap_p0_1", 0)
            print("deltap_p0_2", 0)
            print("epsilon_x1", self.current_emitx_B1)
            print("epsilon_x2", self.current_emitx_B2)
            print("epsilon_y1", self.current_emity_B1)
            print("epsilon_y2", self.current_emity_B2)
            print("sigma_z1", sigma_z1)
            print("sigma_z2", sigma_z2)
            print("beta_x1", betax)
            print("beta_x2", betax)
            print("beta_y1", betay)
            print("beta_y2", betay)
            print("alpha_x1", 0)
            print("alpha_x2", -0)
            print("alpha_y1", 0)
            print("alpha_y2", -0)
            print("dx_1", 0)
            print("dx_2", 0)
            print("dy_1", 0)
            print("dy_2", 0)
            print("dpx_1", 0)
            print("dpx_2", 0)
            print("dpy_1", 0)
            print("dpy_2", 0)
            print("CC_V_x_1", ccvx)
            print("CC_f_x_1", ccrf)
            print("CC_phase_x_1", 0)
            print("CC_V_x_2", -ccvx)
            print("CC_f_x_2", ccrf)
            print("CC_phase_x_2", 0)
            print("CC_V_y_1", ccvy)
            print("CC_f_y_1", ccrf)
            print("CC_phase_y_1", 0)
            print("CC_V_y_2", -ccvy)
            print("CC_f_y_2", ccrf)
            print("CC_phase_y_2", 0)
            print("R12_1", ccr12)
            print("R22_1", 0)
            print("R34_1", ccr34)
            print("R44_1", 0)
            print("R12_2", ccr12)
            print("R22_2", 0)
            print("R34_2", ccr34)
            print("R44_2", 0)
            print("verbose", verbose)
            print("sigma_integration", 3)
            print("")

        if emit is None:
            epsilon_x1=self.current_emitx_B1
            epsilon_x2=self.current_emitx_B2
            epsilon_y1=self.current_emity_B1
            epsilon_y2=self.current_emity_B2
        else:
            epsilon_x1=emit
            epsilon_x2=emit
            epsilon_y1=emit
            epsilon_y2=emit

        if flag == "lumi":

            return lumi_integrals.luminosity(f=self.frev, nb = number_of_bunches, 
                                                    N1 = self.current_inten_B1, N2 =self.current_inten_B2,
                                                    x_1 = x, x_2 = -x, y_1 = y, y_2=-y,
                                                    px_1=px, px_2=-px, py_1=py, py_2=-py,
                                                    energy_tot1 = self.En*1e9, energy_tot2  =self.En*1e9,
                                                    deltap_p0_1=0, deltap_p0_2=0,
                                                    epsilon_x1=epsilon_x1, epsilon_x2=epsilon_x2,
                                                    epsilon_y1=epsilon_y1, epsilon_y2=epsilon_y2,
                                                    sigma_z1=sigma_z1,sigma_z2=sigma_z2,
                                                    beta_x1=betax,beta_x2=betax,beta_y1=betay,beta_y2=betay,
                                                    alpha_x1=0,alpha_x2=-0,alpha_y1=0,alpha_y2=-0,
                                                    dx_1=0,dx_2=0,dy_1=0,dy_2=0,
                                                    dpx_1=0,dpx_2=0,dpy_1=0,dpy_2=0,
                                                    CC_V_x_1=self.cc_b1_fraction*ccvx,CC_f_x_1=ccrf,CC_phase_x_1=0,
                                                    CC_V_x_2=-self.cc_b2_fraction*ccvx,CC_f_x_2=ccrf,CC_phase_x_2=0,
                                                    CC_V_y_1=self.cc_b1_fraction*ccvy,CC_f_y_1=ccrf,CC_phase_y_1=0,
                                                    CC_V_y_2=-self.cc_b2_fraction*ccvy,CC_f_y_2=ccrf,CC_phase_y_2=0,
                                                    R12_1=ccr12,R22_1=0,R34_1=ccr34,R44_1=0,R12_2=ccr12,R22_2=0,R34_2=ccr34,R44_2=0,
                                                    verbose=verbose,sigma_integration=3,
                                                    )
        elif flag == "longitudinal luminous region":
            sigma_z_list = np.linspace(-0.09, 0.09, 100)
            return lumi_integrals.longitudinal_luminous_region(sigma_z_list, f=self.frev, nb = number_of_bunches, 
                                                    N1 = self.current_inten_B1, N2 =self.current_inten_B2,
                                                    x_1 = x, x_2 = -x, y_1 = y, y_2=-y,
                                                    px_1=px, px_2=-px, py_1=py, py_2=-py,
                                                    energy_tot1 = self.En*1e9, energy_tot2  =self.En*1e9,
                                                    deltap_p0_1=0, deltap_p0_2=0,
                                                    epsilon_x1=epsilon_x1, epsilon_x2=epsilon_x2,
                                                    epsilon_y1=epsilon_y1, epsilon_y2=epsilon_y2,
                                                    sigma_z1=sigma_z1,sigma_z2=sigma_z2,
                                                    beta_x1=betax,beta_x2=betax,beta_y1=betay,beta_y2=betay,
                                                    alpha_x1=0,alpha_x2=-0,alpha_y1=0,alpha_y2=-0,
                                                    dx_1=0,dx_2=0,dy_1=0,dy_2=0,
                                                    dpx_1=0,dpx_2=0,dpy_1=0,dpy_2=0,
                                                    CC_V_x_1=self.cc_b1_fraction*ccvx,CC_f_x_1=ccrf,CC_phase_x_1=0,
                                                    CC_V_x_2=-self.cc_b2_fraction*ccvx,CC_f_x_2=ccrf,CC_phase_x_2=0,
                                                    CC_V_y_1=self.cc_b1_fraction*ccvy,CC_f_y_1=ccrf,CC_phase_y_1=0,
                                                    CC_V_y_2=-self.cc_b2_fraction*ccvy,CC_f_y_2=ccrf,CC_phase_y_2=0,
                                                    R12_1=ccr12,R22_1=0,R34_1=ccr34,R44_1=0,R12_2=ccr12,R22_2=0,R34_2=ccr34,R44_2=0,
                                                    verbose=verbose,sigma_integration=3,
                                                    )



    def cmp_pileup(self, luminosity, how="ip15"):
        if how == "ip15":
            return luminosity*self.sigmaBOff_m2/(self.number_of_bunches_coll_ip15*self.frev)
        elif how == "ip8":
            return luminosity*self.sigmaBOff_m2/(self.number_of_bunches_coll_ip8*self.frev)
        else:
            raise ValueError("how should be 'ip15' or 'ip8'")

    def compute_CMS_ATLAS_lumi(self, flag = "lumi", **kwargs):

        current_inten_B1, current_inten_B2 = self.current_inten_B1, self.current_inten_B2
        current_emitx_B1, current_emity_B1 = self.current_emitx_B1, self.current_emity_B1
        current_emitx_B2, current_emity_B2 = self.current_emitx_B2, self.current_emity_B2
        current_bl_s_B1, current_bl_s_B2 = self.current_bl_s_B1, self.current_bl_s_B2
        current_betax_ip1, current_betay_ip1 = self.current_betax_ip1, self.current_betay_ip1
        current_betax_ip5, current_betay_ip5 = self.current_betax_ip5, self.current_betay_ip5
        current_xing_ip1_x, current_xing_ip1_y = self.current_xing_ip1_x, self.current_xing_ip1_y
        current_xing_ip5_x, current_xing_ip5_y = self.current_xing_ip5_x, self.current_xing_ip5_y
        current_on_sep1_x, current_on_sep1_y = self.current_on_sep1_x, self.current_on_sep1_y
        current_on_sep5_x, current_on_sep5_y = self.current_on_sep5_x, self.current_on_sep5_y
        verbose = False

        if "current_inten_B1" in kwargs:
            current_inten_B1 = kwargs["current_inten_B1"]
        if "current_inten_B2" in kwargs:
            current_inten_B2 = kwargs["current_inten_B2"]
        if "current_emitx_B1" in kwargs:
            current_emitx_B1 = kwargs["current_emitx_B1"]
        if "current_emity_B1" in kwargs:
            current_emity_B1 = kwargs["current_emity_B1"]
        if "current_emitx_B2" in kwargs:
            current_emitx_B2 = kwargs["current_emitx_B2"]
        if "current_emity_B2" in kwargs:
            current_emity_B2 = kwargs["current_emity_B2"]
        if "current_bl_s_B1" in kwargs:
            current_bl_s_B1 = kwargs["current_bl_s_B1"]
        if "current_bl_s_B2" in kwargs:
            current_bl_s_B2 = kwargs["current_bl_s_B2"]
        if "current_betax_ip1" in kwargs:
            current_betax_ip1 = kwargs["current_betax_ip1"]
        if "current_betay_ip1" in kwargs:
            current_betay_ip1 = kwargs["current_betay_ip1"]
        if "current_betax_ip5" in kwargs:
            current_betax_ip5 = kwargs["current_betax_ip5"]
        if "current_betay_ip5" in kwargs:
            current_betay_ip5 = kwargs["current_betay_ip5"]
        if "current_xing_ip1_x" in kwargs:
            current_xing_ip1_x = kwargs["current_xing_ip1_x"]
        if "current_xing_ip1_y" in kwargs:
            current_xing_ip1_y = kwargs["current_xing_ip1_y"]
        if "current_xing_ip5_x" in kwargs:
            current_xing_ip5_x = kwargs["current_xing_ip5_x"]
        if "current_xing_ip5_y" in kwargs:
            current_xing_ip5_y = kwargs["current_xing_ip5_y"]
        if "current_on_sep1_x" in kwargs:
            current_on_sep1_x = kwargs["current_on_sep1_x"]
        if "current_on_sep1_y" in kwargs:
            current_on_sep1_y = kwargs["current_on_sep1_y"]
        if "current_on_sep5_x" in kwargs:
            current_on_sep5_x = kwargs["current_on_sep5_x"]
        if "current_on_sep5_y" in kwargs:
            current_on_sep5_y = kwargs["current_on_sep5_y"]
        if "verbose" in kwargs:
            verbose = kwargs["verbose"]
        

        if self.ATLAS_crossing == "V":
            myxing_ip1 = current_xing_ip1_y
            #print(myxing)
            current_xing_ip1_x = 0
            current_on_sep1_y = 0
        else:
            current_xing_ip1_y = 0
            myxing_ip1 = current_xing_ip1_x
            current_on_sep1_x = 0
        if self.CMS_crossing == "V":
            myxing_ip5 = current_xing_ip5_y
            current_xing_ip5_x = 0
            current_on_sep5_y = 0
        else:
            current_xing_ip5_y = 0
            myxing_ip5 = current_xing_ip5_x
            current_on_sep5_x = 0

        if verbose:
            print("xing_ip1_x", current_xing_ip1_x)
            print("xing_ip1_y", current_xing_ip1_y)
            print("xing_ip5_x", current_xing_ip5_x)
            print("xing_ip5_y", current_xing_ip5_y)
            print("myxing_ip1", myxing_ip1)
            print("myxing_ip5", myxing_ip5)
            print("on_sep1_x", current_on_sep1_x)
            print("on_sep1_y", current_on_sep1_y)
            print("on_sep5_x", current_on_sep5_x)
            print("on_sep5_y", current_on_sep5_y)

        if flag == "lumi":
            if self.flag_lumi_simple:

                current_lumi_ATLAS = self.Lumi_inst_IP15(current_inten_B1, current_inten_B2,
                                                        current_emitx_B1, current_emity_B1,
                                                        current_emitx_B2, current_emity_B2,
                                                        current_bl_s_B1, current_bl_s_B2,
                                                        current_betax_ip1, current_betay_ip1, myxing_ip1, "ATLAS", sepx=current_on_sep1_x, sepy=current_on_sep1_y)
                current_lumi_CMS = self.Lumi_inst_IP15(current_inten_B1, current_inten_B2,
                                                        current_emitx_B1, current_emity_B1,
                                                        current_emitx_B2, current_emity_B2,
                                                        current_bl_s_B1, current_bl_s_B2,
                                                        current_betax_ip5, current_betay_ip5, myxing_ip5, "CMS", sepx=current_on_sep5_x, sepy=current_on_sep5_y)
            else:
                
                current_lumi_ATLAS = self.Lumi_integrals(self.number_of_bunches_coll_ip15, betax = current_betax_ip1, betay= current_betay_ip1, x=current_on_sep1_x, px=current_xing_ip1_x/2., y=current_on_sep1_y, py=current_xing_ip1_y/2., ccx=self.ccx_ip1, ccy=self.ccy_ip1, ccrf=400e6,verbose=False)
                self.current_ccvx_ip1 = self.ccvx
                self.current_ccvy_ip1 = self.ccvy
                current_lumi_CMS = self.Lumi_integrals(self.number_of_bunches_coll_ip15, betax = current_betax_ip5, betay= current_betay_ip5, x=current_on_sep5_x, px=current_xing_ip5_x/2., y=current_on_sep5_y, py=current_xing_ip5_y/2., ccx=self.ccx_ip5, ccy=self.ccy_ip5, ccrf=400e6,verbose=False)
                self.current_ccvx_ip5 = self.ccvx
                self.current_ccvy_ip5 = self.ccvy
            return current_lumi_ATLAS, current_lumi_CMS
        elif flag == "longitudinal luminous region":
            longitudinal_density_ATLAS, current_longitudinal_luminous_region_ATLAS, FWHM_ALTAS = self.Lumi_integrals(self.number_of_bunches_coll_ip15, betax = current_betax_ip1, betay= current_betay_ip1, x=current_on_sep1_x, px=current_xing_ip1_x/2., y=current_on_sep1_y, py=current_xing_ip1_y/2., ccx=self.ccx_ip1, ccy=self.ccy_ip1, ccrf=400e6,verbose=verbose, flag=flag)
            longitudinal_density_CMS, current_longitudinal_luminous_region_CMS, FWHM_CMS = self.Lumi_integrals(self.number_of_bunches_coll_ip15, betax = current_betax_ip5, betay= current_betay_ip5, x=current_on_sep5_x, px=current_xing_ip5_x/2., y=current_on_sep5_y, py=current_xing_ip5_y/2., ccx=self.ccx_ip5, ccy=self.ccy_ip5, ccrf=400e6,verbose=verbose, flag=flag)           
            return current_longitudinal_luminous_region_ATLAS, current_longitudinal_luminous_region_CMS


    ########################################################################################################
    ########################################################################################################
    # IBS useful functions
        
    def func_load_lookup_table(self):
        
        # Function that loads the lookup table, does the reshuffling of the lookup tables and performs the 3D interpolation 
        # Returns a dictionary with the equilibrium properties and the interpolated functions
        # The interpolation is done for each intensity, for completeness. Howevere, in practice only 1 intensity is used and 
        # a scaling with intensity is done, as the growth rates scale linearly with intensity. 
        
        self.lookup_table = {}
        for key in self.lookup_table_path.keys():   
            dict_table = pd.read_parquet(self.lookup_table_path[key])
            

            if not self.flag_lookup_xibs:
                # Extract values directly, assuming first row contains relevant data
                ex0, ey0, sp0, ss0, taux, tauy, taul, En = dict_table.iloc[0][['ex0', 'ey0', 'sp0', 'ss0', 'taux', 'tauy', 'taul', 'en']]
            else:
                # Direct extraction and computation where possible
                ex0, ey0, sp0, ss0 = dict_table.iloc[0][['eq_nemit_x', 'eq_nemit_y', 'eq_nemit_z', 'eq_nemit_z']].values
                taux, tauy, taul = dict_table[['taux_h_SR', 'tauy_h_SR', 'taul_h_SR']].values[0] * 3600.0
                
                # Pre-compute and assign new columns to avoid repeating these operations
                dict_table['npbb'] = dict_table["n_part"]
                dict_table['exin'] = dict_table["norm_epsx"] * 1e6
                dict_table['eyin'] = dict_table["norm_epsy"] * 1e6
                dict_table['bl_ns'] = dict_table["bunch_length"] * 4 / clight * 1e9
                
                # Unique computations done once
                exun, eyun, blun, Nbun = [np.unique(dict_table[col]) for col in ['exin', 'eyin', 'bl_ns', 'npbb']]

                dict_table['npbb'] = dict_table["n_part"]
                dict_table['exin'] = dict_table["norm_epsx"]*1e6
                dict_table['eyin'] = dict_table["norm_epsy"]*1e6
                dict_table['bl_ns'] = dict_table["bunch_length"]*4/clight*1e9
                dict_table['txh'] = dict_table["taux_h_IBS"]
                dict_table['tyh'] = dict_table["tauy_h_IBS"]
                dict_table['tlh'] = dict_table["taul_h_IBS"]


            myarg=0
            
            dict_interp = {'equilibrium_properties': {},'Nb':{}}
        for nn, Nbb in enumerate(Nbun[0:1]):
            dict_interp['Nb'][Nbb] = {}

            mask_nb = dict_table['npbb'] == Nbb
            df_masked = dict_table[mask_nb]

            # Precompute a 3D grid for each parameter
            Tx = np.full((len(exun), len(eyun), len(blun)), np.nan)
            Ty = np.full((len(exun), len(eyun), len(blun)), np.nan)
            Tl = np.full((len(exun), len(eyun), len(blun)), np.nan)

            # Optimize by reducing the use of np.isclose in a loop
            # Assign values to Tx, Ty, Tl using direct indexing when possible
            for index, row in df_masked.iterrows():
                ex_index = np.where(np.round(exun, 5) == np.round(row['exin'], 5))[0]
                ey_index = np.where(np.round(eyun, 5) == np.round(row['eyin'], 5))[0]
                bl_index = np.where(np.round(blun, 5) == np.round(row['bl_ns'], 5))[0]
                
                # If any of the parameters are not found in the unique arrays, skip this iteration
                if not (ex_index.size and ey_index.size and bl_index.size):
                    continue
                
                # Fill the grids with the corresponding values
                Tx[ex_index[0], ey_index[0], bl_index[0]] = row['txh']
                Ty[ex_index[0], ey_index[0], bl_index[0]] = row['tyh']
                Tl[ex_index[0], ey_index[0], bl_index[0]] = row['tlh']

            # Handle NaNs in Tx, Ty, Tl if necessary, e.g., by setting them to zero or using interpolation to fill gaps

            # Create the RegularGridInterpolator objects
            fnx = rgi((exun, eyun, blun), Tx, bounds_error=False, fill_value=None)
            fny = rgi((exun, eyun, blun), Ty, bounds_error=False, fill_value=None)
            fnl = rgi((exun, eyun, blun), Tl, bounds_error=False, fill_value=None)

            # Store the interpolator functions
            dict_interp['Nb'][Nbb]['txh'] = fnx
            dict_interp['Nb'][Nbb]['tyh'] = fny
            dict_interp['Nb'][Nbb]['tlh'] = fnl
            dict_interp['equilibrium_properties']['ex0'] = ex0
            dict_interp['equilibrium_properties']['ey0'] = ey0
            dict_interp['equilibrium_properties']['sp0'] = sp0
            dict_interp['equilibrium_properties']['ss0'] = ss0
            dict_interp['equilibrium_properties']['taux'] = taux
            dict_interp['equilibrium_properties']['tauy'] = tauy
            dict_interp['equilibrium_properties']['taul'] = taul

        self.lookup_table[key] = dict_interp
    
    
    def growth_rates(self, npb, exi, eyi, bl_ns):
        try:
                #Nbkeys = Nbun
                Nbkeys = list(self.current_lookup_table['Nb'].keys())
                argn = [0]
                taux_h = (self.current_lookup_table['Nb'][Nbkeys[argn[0]]]['txh'])((exi,eyi,bl_ns))*Nbkeys[argn[0]]/npb
                tauy_h = (self.current_lookup_table['Nb'][Nbkeys[argn[0]]]['tyh'])((exi,eyi,bl_ns))*Nbkeys[argn[0]]/npb
                taul_h = (self.current_lookup_table['Nb'][Nbkeys[argn[0]]]['tlh'])((exi,eyi,bl_ns))*Nbkeys[argn[0]]/npb
        except:
                print(f'Error on growth rates calculation. Check if out of range of the lookup table')
                print(f'ex={min(exi), max(exi)}, ey={min(eyi), max(eyi)}, bl_ns={min(bl_ns), max(bl_ns)}')
                taux_h = tauy_h = taul_h = np.nan
        return taux_h, tauy_h, taul_h
    
    ########################################################################################################
    ########################################################################################################
    
    ########################################################################################################
    ########################################################################################################
    # Models for emit growth

    def SR_collider(self):
        import xtrack as xt

        collider = xt.Multiline.from_json(self.collider_path)
        self.twiss = collider.lhcb1.twiss()
        collider.lhcb1.configure_radiation(model='mean')
        self.twiss_SR = collider.lhcb1.twiss(eneloss_and_damping=True)
        #beam_params = BeamParameters(twiss["particle_on_co"])
        #optics = OpticsParameters(twiss)

        U0_eV = self.twiss_SR['eneloss_turn']
        self.tauSRxy_s=2*self.En*1e9*(self.circ/clight)/U0_eV;
        self.tauSRl_s=2*self.En*1e9*(self.circ/clight)/U0_eV/2;
        self.U0 = U0_eV/1e9

    def IBS_model(self, emitx_B1_for_growth_rates, emitx_B1_apply_IBS,
                  emity_B1_for_growth_rates, emity_B1_apply_IBS,
                  emitx_B2_for_growth_rates, emitx_B2_apply_IBS,
                  emity_B2_for_growth_rates, emity_B2_apply_IBS):
        
        bli = self.current_bl_s_B1*clight/4
        dpp=np.sqrt(2/np.pi)*np.sqrt(self.nc * (self.V0 - self.U0) * (np.sin(bli * self.h * np.pi * self.betar/self.circ))**2)/np.sqrt(self.En*self.h*np.abs(self.etap))/self.betar

        if not self.flag_IBS_xibs:
            ex0 = self.current_lookup_table['equilibrium_properties']['ex0']
            ey0 = self.current_lookup_table['equilibrium_properties']['ey0']
            sp0 = self.current_lookup_table['equilibrium_properties']['sp0']
            ss0 = self.current_lookup_table['equilibrium_properties']['ss0']
            taux = self.current_lookup_table['equilibrium_properties']['taux']
            tauy = self.current_lookup_table['equilibrium_properties']['tauy']
            taul = self.current_lookup_table['equilibrium_properties']['taul']

            ## For Beam 1

            #taux_h, tauy_h, taul_h = self.growth_rates(self.current_inten_B1, self.current_emitx_B1_uncoupled*1e6, self.current_emity_B1_uncoupled*1e6, self.current_bl_s_B1*1e9) 
            taux_h, tauy_h, taul_h = self.growth_rates(self.current_inten_B1, emitx_B1_for_growth_rates*1e6, emity_B1_for_growth_rates*1e6, self.current_bl_s_B1*1e9)
        else:
            ex0 = self.twiss_SR["eq_nemitt_x"]
            ey0 = self.twiss_SR["eq_nemitt_y"]
            sp0 = self.twiss_SR["eq_nemitt_zeta"]
            taux = self.tauSRxy_s
            tauy = self.tauSRxy_s
            taul = self.tauSRl_s
            from xibs.analytical import BjorkenMtingwaIBS
            from xibs.inputs import BeamParameters, OpticsParameters

            beam_params = BeamParameters(self.twiss["particle_on_co"])
            optics = OpticsParameters(self.twiss)

            results = [
                (setattr(beam_params, 'n_part', self.current_inten_B1[i]),  # Update beam_params.n_part
                BjorkenMtingwaIBS(beam_params, optics).growth_rates(  # Instantiate and call growth_rates
                    self.current_emitx_B1_uncoupled[i], 
                    self.current_emity_B1_uncoupled[i], 
                    np.sqrt(2/np.pi)*np.sqrt(self.nc * (self.V0 - self.U0) * (np.sin(self.current_bl_s_B1[i]*clight/4 * self.h * np.pi * self.betar/self.circ))**2)/np.sqrt(self.En*self.h*np.abs(self.etap))/self.betar, 
                    self.current_bl_s_B1[i]*clight/4, 
                    normalized_emittances=True
                ))[1]  # Select only the growth_rates result, discarding the result of setattr
                for i in range(len(self.current_inten_B1))
            ]

            taux_h = np.array([1.0 / result.Tx / 3600.0 for result in results])
            tauy_h = np.array([1.0 / result.Ty / 3600.0 for result in results])
            taul_h = np.array([1.0 / result.Tz / 3600.0 for result in results])


            
        self.taux_h = taux_h
        self.tauy_h = tauy_h
        self.taul_h = taul_h
        self.current_Tx_B1 = 1./(taux_h*3600.)/2.
        self.current_Ty_B1 = 1./(tauy_h*3600.)/2.
        self.current_Tl_B1 = 1./(taul_h*3600.)/2.

        #ex_out_norm_m = (-ex0+np.exp(2*self.dt_step*(self.current_Tx_B1-1/taux))*(ex0+self.current_emitx_B1_uncoupled*(-1+self.current_Tx_B1*taux)))/(-1+self.current_Tx_B1*taux)
        #ey_out_norm_m = (-ey0+np.exp(2*self.dt_step*(self.current_Ty_B1-1/tauy))*(ey0+self.current_emity_B1_uncoupled*(-1+self.current_Ty_B1*tauy)))/(-1+self.current_Ty_B1*tauy)
        ex_out_norm_m = (-ex0+np.exp(2*self.dt_step*(self.current_Tx_B1-1/taux))*(ex0+emitx_B1_apply_IBS*(-1+self.current_Tx_B1*taux)))/(-1+self.current_Tx_B1*taux)
        ey_out_norm_m = (-ey0+np.exp(2*self.dt_step*(self.current_Ty_B1-1/tauy))*(ey0+emity_B1_apply_IBS*(-1+self.current_Ty_B1*tauy)))/(-1+self.current_Ty_B1*tauy)
        dpp = (-sp0+np.exp(self.dt_step*(self.current_Tl_B1-1/taul))*(sp0+dpp*(-1+self.current_Tl_B1*taul)))/(-1+self.current_Tl_B1*taul)


        bl_out = clight*self.circ*np.arccos((self.En*self.nc*(self.V0-self.U0)*self.betar**2-dpp**2*self.En**2*self.h*np.pi*self.betar**4*abs(self.etap))/(self.En*self.nc*(self.V0-self.U0)*self.betar**2))/(2*clight*self.h*np.pi*self.betar)
        bl_out_4sigma_s = bl_out/clight*4
        
        self.current_emitx_B1_IBS = ex_out_norm_m
        self.current_emity_B1_IBS = ey_out_norm_m
        self.current_bl_s_B1_IBS  = bl_out_4sigma_s

        if np.mean(self.current_bl_s_B1_IBS) < self.level_bl_at:
            self.current_bl_s_B1_IBS = np.array([self.level_bl_to]*len(self.current_bl_s_B1_IBS))

        bli = self.current_bl_s_B2*clight/4
        dpp=np.sqrt(2/np.pi)*np.sqrt(self.nc * (self.V0 - self.U0) * (np.sin(bli * self.h * np.pi * self.betar/self.circ))**2)/np.sqrt(self.En*self.h*np.abs(self.etap))/self.betar


        ## For Beam 2
        if not self.flag_IBS_xibs:
            #bli = self.current_bl_s_B2*clight/4
            #dpp=np.sqrt(2/np.pi)*np.sqrt(self.nc * (self.V0 - self.U0) * (np.sin(bli * self.h * np.pi * self.betar/self.circ))**2)/np.sqrt(self.En*self.h*np.abs(self.etap))/self.betar
            #taux_h, tauy_h, taul_h = self.growth_rates(self.current_inten_B2, self.current_emitx_B2_uncoupled*1e6, self.current_emity_B2_uncoupled*1e6, self.current_bl_s_B2*1e9)
            taux_h, tauy_h, taul_h = self.growth_rates(self.current_inten_B2, emitx_B2_for_growth_rates*1e6, emity_B2_for_growth_rates*1e6, self.current_bl_s_B2*1e9)
        else:
            results = [
                (setattr(beam_params, 'n_part', self.current_inten_B2[i]),  # Update beam_params.n_part
                BjorkenMtingwaIBS(beam_params, optics).growth_rates(  # Instantiate and call growth_rates
                    self.current_emitx_B2_uncoupled[i], 
                    self.current_emity_B2_uncoupled[i], 
                    np.sqrt(2/np.pi)*np.sqrt(self.nc * (self.V0 - self.U0) * (np.sin(self.current_bl_s_B2[i]*clight/4 * self.h * np.pi * self.betar/self.circ))**2)/np.sqrt(self.En*self.h*np.abs(self.etap))/self.betar, 
                    self.current_bl_s_B2[i]*clight/4, 
                    normalized_emittances=True
                ))[1]  # Select only the growth_rates result, discarding the result of setattr
                for i in range(len(self.current_inten_B2))
            ]

            taux_h = np.array([1.0 / result.Tx / 3600.0 for result in results])
            tauy_h = np.array([1.0 / result.Ty / 3600.0 for result in results])
            taul_h = np.array([1.0 / result.Tz / 3600.0 for result in results])
            
        self.current_Tx_B2 = 1./(taux_h*3600.)/2.
        self.current_Ty_B2 = 1./(tauy_h*3600.)/2.
        self.current_Tl_B2 = 1./(taul_h*3600.)/2.

        #ex_out_norm_m = (-ex0+np.exp(2*self.dt_step*(self.current_Tx_B2-1/taux))*(ex0+self.current_emitx_B2_uncoupled*(-1+self.current_Tx_B2*taux)))/(-1+self.current_Tx_B2*taux)
        #ey_out_norm_m = (-ey0+np.exp(2*self.dt_step*(self.current_Ty_B2-1/tauy))*(ey0+self.current_emity_B2_uncoupled*(-1+self.current_Ty_B2*tauy)))/(-1+self.current_Ty_B2*tauy)
        ex_out_norm_m = (-ex0+np.exp(2*self.dt_step*(self.current_Tx_B2-1/taux))*(ex0+emitx_B2_apply_IBS*(-1+self.current_Tx_B2*taux)))/(-1+self.current_Tx_B2*taux)
        ey_out_norm_m = (-ey0+np.exp(2*self.dt_step*(self.current_Ty_B2-1/tauy))*(ey0+emity_B2_apply_IBS*(-1+self.current_Ty_B2*tauy)))/(-1+self.current_Ty_B2*tauy)
        dpp = (-sp0+np.exp(self.dt_step*(self.current_Tl_B2-1/taul))*(sp0+dpp*(-1+self.current_Tl_B2*taul)))/(-1+self.current_Tl_B2*taul)

        bl_out = clight*self.circ*np.arccos((self.En*self.nc*(self.V0-self.U0)*self.betar**2-dpp**2*self.En**2*self.h*np.pi*self.betar**4*abs(self.etap))/(self.En*self.nc*(self.V0-self.U0)*self.betar**2))/(2*clight*self.h*np.pi*self.betar)
        bl_out_4sigma_s = bl_out/clight*4

        self.current_emitx_B2_IBS = ex_out_norm_m
        self.current_emity_B2_IBS = ey_out_norm_m
        self.current_bl_s_B2_IBS  = bl_out_4sigma_s

        if np.mean(self.current_bl_s_B2_IBS) < self.level_bl_at:
            self.current_bl_s_B2_IBS = np.array([self.level_bl_to]*len(self.current_bl_s_B2_IBS))

        
    def SR_model(self, emity_B1_apply_SR, emity_B2_apply_SR):
        self.current_emity_B1_IBS_SR = emity_B1_apply_SR*np.exp(-2*self.dt_step/self.tauSRxy_s)
        self.current_emity_B2_IBS_SR = emity_B2_apply_SR*np.exp(-2*self.dt_step/self.tauSRxy_s)

    def Elastic_scattering_model(self, nIP=2):
        theta = np.sqrt((1./19.9)/self.En**2)
        theta_x = theta/np.sqrt(2.)
        L = 1e-4*0.5*(self.current_lumi_CMS + self.current_lumi_ATLAS)/self.number_of_bunches_coll_ip15

        den_x_dt = 0.5*nIP*self.gamma*self.current_betax*100*theta_x**2 * L * self.sigma_el_m2*1e4/self.current_inten_B1
        den_x_dt_mmmradpersec = den_x_dt*10*1000  
        den_y_dt = 0.5*nIP*self.gamma*self.current_betay*100*theta_x**2 * L * self.sigma_el_m2*1e4/self.current_inten_B1
        den_y_dt_mmmradpersec = den_y_dt*10*1000 

        den_dt = 0.5 * nIP * self.gamma*self.current_betax*100 * theta**2 * L * self.sigma_el_m2*1e4/self.current_inten_B1
        den_dt_mmmradpersec = den_dt*10*1000 

        self.dexdt1, self.deydt1, self.deconvdt1 = den_x_dt_mmmradpersec*1e-6, den_y_dt_mmmradpersec*1e-6, den_dt_mmmradpersec*1e-6

        den_x_dt = 0.5*nIP*self.gamma*self.current_betax*100*theta_x**2 * L * self.sigma_el_m2*1e4/self.current_inten_B2
        den_x_dt_mmmradpersec = den_x_dt*10*1000 
        den_y_dt = 0.5*nIP*self.gamma*self.current_betay*100*theta_x**2 * L * self.sigma_el_m2*1e4/self.current_inten_B2
        den_y_dt_mmmradpersec = den_y_dt*10*1000 

        den_dt = 0.5 * nIP * self.gamma*self.current_betax*100 * theta**2 * L * self.sigma_el_m2*1e4/self.current_inten_B2
        den_dt_mmmradpersec = den_dt*10*1000 

        self.dexdt2, self.deydt2, self.deconvdt2 = den_x_dt_mmmradpersec*1e-6, den_y_dt_mmmradpersec*1e-6, den_dt_mmmradpersec*1e-6

    def Coupling_model(self, emitx_B1, emity_B1, emitx_B2, emity_B2, verbose=False):
        # Coupling

        coupling_coeff = self.Coupling_coeff
        delta_ts = self.Delta_tunesplit
                                
        self.Cfactor = coupling_coeff**2/2/(delta_ts**2+coupling_coeff**2)
        #self.dexy1   = self.current_emitx_B1_uncoupled - self.current_emity_B1_uncoupled
        #self.dexy2   = self.current_emitx_B2_uncoupled - self.current_emity_B2_uncoupled
        self.dexy1   = emitx_B1 - emity_B1
        self.dexy2   = emitx_B2 - emity_B2
        #if verbose:
        #    print(self.Cfactor,self.dexy1,self.dexy2, self.current_emitx_B1_uncoupled, self.current_emity_B1_uncoupled, self.current_emitx_B2_uncoupled, self.current_emity_B2_uncoupled)

    def ExtraBU_model(self):        
        egrx = self.dt_step*self.ex_extra/3600*1e-6; #!0.05 um/h extra emittance blow up in horizontal
        egry = self.dt_step*self.ey_extra/3600*1e-6; #!0.05 um/h extra emittance blow up in vertical

        #self.current_emitx_B1_IBS     = self.current_emitx_B1_IBS + egrx
        #self.current_emity_B1_IBS_SR  = self.current_emity_B1_IBS_SR  + egry
        #self.current_emitx_B2_IBS     = self.current_emitx_B2_IBS + egrx
        #self.current_emity_B2_IBS_SR  = self.current_emity_B2_IBS_SR + egry
        self.current_emitx_B1_uncoupled += egrx
        self.current_emity_B1_uncoupled += egry
        self.current_emitx_B2_uncoupled += egrx
        self.current_emity_B2_uncoupled += egry
        self.current_emitx_B1 += egrx
        self.current_emity_B1 += egry
        self.current_emitx_B2 += egrx
        self.current_emity_B2 += egry


    def CCnoise_model(self, verbose=False):
        # 0.115
        egrx = self.dt_step*self.ex_extra_CC/3600*1e-6; 
        egry = self.dt_step*self.ey_extra_CC/3600*1e-6; 

        current_beta = max(self.current_betax_ip1, self.current_betay_ip1)
        factor = (self.VCC_MV/6.8)**2 * 0.15/current_beta
        #print("Factor: ", factor)
        egrx = egrx*factor
        egry = egry*factor

        f1 = self.cc_b1_fraction**2
        f2 = self.cc_b2_fraction**2
        self.current_emitx_B1_uncoupled += f1*egrx
        self.current_emity_B1_uncoupled += f1*egry
        self.current_emitx_B2_uncoupled += f2*egrx
        self.current_emity_B2_uncoupled += f2*egry
        self.current_emitx_B1 += f1*egrx
        self.current_emity_B1 += f1*egry
        self.current_emitx_B2 += f2*egrx
        self.current_emity_B2 += f2*egry

    
    ########################################################################################################
    ########################################################################################################
    
    ########################################################################################################
    ########################################################################################################

    def Boff_losses_model(self, verbose=False):
            
            lumi_nolhcb = self.current_lumi_ATLAS/self.number_of_bunches_coll_ip15 + self.current_lumi_CMS/self.number_of_bunches_coll_ip15

            tauBOff_s1_nolhcb = self.current_inten_B1/(self.sigmaBOff_losses * lumi_nolhcb )
            tauBOff_s2_nolhcb = self.current_inten_B2/(self.sigmaBOff_losses * lumi_nolhcb)

            if self.include_lhcb:
                if self.verbose or verbose:
                    print("LHCB included")
                lumi_withlhcb = self.current_lumi_ATLAS/self.number_of_bunches_coll_ip15 + self.current_lumi_CMS/self.number_of_bunches_coll_ip15 + self.current_lumi_LHCB/self.number_of_bunches_coll_ip8
                tauBOff_s1_withlhcb = self.current_inten_B1/(self.sigmaBOff_losses * lumi_withlhcb )
                tauBOff_s2_withlhcb = self.current_inten_B2/(self.sigmaBOff_losses * lumi_withlhcb)
                tauBOff_s1 = tauBOff_s1_withlhcb
                tauBOff_s2 = tauBOff_s2_withlhcb
            else:
                if self.verbose or verbose:
                    print("LHCB NOT included")
                tauBOff_s1 = tauBOff_s1_nolhcb
                tauBOff_s2 = tauBOff_s2_nolhcb
            self.current_inten_B1 = self.current_inten_B1/(1.+self.dt_step/tauBOff_s1)
            self.current_inten_B2 = self.current_inten_B2/(1.+self.dt_step/tauBOff_s2)
            #if self.verbose or verbose:
            if (self.current_time <20) and self.include_lhcb:
                print(tauBOff_s1/3600.0, tauBOff_s2/3600.0, tauBOff_s1_nolhcb/3600.0, tauBOff_s2_nolhcb/3600.0, tauBOff_s1_withlhcb/3600.0, tauBOff_s2_withlhcb/3600.0)
                print("Equivalent burn off cross section from LHCB: ", (lumi_withlhcb/lumi_nolhcb * tauBOff_s1_nolhcb/tauBOff_s1_withlhcb *self.sigmaBOff_losses*1e31 -self.sigmaBOff_losses*1e31))


    ########################################################################################################
    ########################################################################################################

    ########################################################################################################
    ########################################################################################################
    # Save results and dump to df

    def save_data(self):
        if self.verbose:
            self.print_current_values()
        self.results_betax[self.current_time] = self.current_betax
        self.results_betay[self.current_time] = self.current_betay
        self.results_betax_ip1[self.current_time] = self.current_betax_ip1
        self.results_betay_ip1[self.current_time] = self.current_betay_ip1
        self.results_betax_ip5[self.current_time] = self.current_betax_ip5
        self.results_betay_ip5[self.current_time] = self.current_betay_ip5
        
        self.results_xing[self.current_time] = self.current_xing
        self.results_xing_ip1_x[self.current_time] = self.current_xing_ip1_x
        self.results_xing_ip1_y[self.current_time] = self.current_xing_ip1_y
        self.results_xing_ip5_x[self.current_time] = self.current_xing_ip5_x
        self.results_xing_ip5_y[self.current_time] = self.current_xing_ip5_y
        self.results_on_sep1_x[self.current_time] = self.current_on_sep1_x
        self.results_on_sep1_y[self.current_time] = self.current_on_sep1_y
        self.results_on_sep5_x[self.current_time] = self.current_on_sep5_x
        self.results_on_sep5_y[self.current_time] = self.current_on_sep5_y
        self.results_emitx_B1[self.current_time] = self.current_emitx_B1
        self.results_emity_B1[self.current_time] = self.current_emity_B1
        self.results_emitx_B2[self.current_time] = self.current_emitx_B2
        self.results_emity_B2[self.current_time] = self.current_emity_B2
        self.results_emitx_B1_uncoupled[self.current_time] = self.current_emitx_B1_uncoupled
        self.results_emity_B1_uncoupled[self.current_time] = self.current_emity_B1_uncoupled
        self.results_emitx_B2_uncoupled[self.current_time] = self.current_emitx_B2_uncoupled
        self.results_emity_B2_uncoupled[self.current_time] = self.current_emity_B2_uncoupled
        self.results_bl_s_B1[self.current_time] = self.current_bl_s_B1
        self.results_bl_s_B2[self.current_time] = self.current_bl_s_B2
        self.results_inten_B1[self.current_time] = self.current_inten_B1
        self.results_inten_B2[self.current_time] = self.current_inten_B2
        self.results_lumi_ATLAS[self.current_time] = self.current_lumi_ATLAS
        self.results_lumi_CMS[self.current_time] = self.current_lumi_CMS
        self.results_lumi_avg[self.current_time] = np.mean([np.mean(self.current_lumi_ATLAS), np.mean(self.current_lumi_CMS)])
        #self.results_longitudinal_luminous_region_ATLAS[self.current_time] = self.current_longitudinal_luminous_region_ATLAS
        #self.results_longitudinal_luminous_region_CMS[self.current_time] = self.current_longitudinal_luminous_region_CMS
        #self.results_longitudinal_luminous_region_avg[self.current_time] = np.mean([np.mean(self.current_longitudinal_luminous_region_ATLAS), np.mean(self.current_longitudinal_luminous_region_CMS)])
        self.results_pu_ATLAS[self.current_time] = self.current_pu_ATLAS
        self.results_pu_CMS[self.current_time] = self.current_pu_CMS
        self.results_pu_avg[self.current_time] = np.mean([np.mean(self.current_pu_ATLAS), np.mean(self.current_pu_CMS)])

        try:

            self.results_ccvx_ip1 = self.current_ccvx_ip1
            self.results_ccvy_ip1 = self.current_ccvy_ip1
            self.results_ccvx_ip5 = self.current_ccvx_ip5
            self.results_ccvy_ip5 = self.current_ccvy_ip5 
        except:
            self.results_ccvx_ip1 = 0
            self.results_ccvy_ip1 = 0
            self.results_ccvx_ip5 = 0
            self.results_ccvy_ip5 = 0

        if self.include_lhcb:
            self.results_lumi_LHCB[self.current_time] = self.current_lumi_LHCB
            self.results_pu_LHCB[self.current_time] = self.current_pu_LHCB
            self.results_betax_ip8[self.current_time] = self.current_betax_ip8
            self.results_betay_ip8[self.current_time] = self.current_betay_ip8
            self.results_xing_ip8_x[self.current_time] = self.current_xing_ip8_x
            self.results_xing_ip8_y[self.current_time] = self.current_xing_ip8_y
            self.results_xing_ip8_r[self.current_time] = self.current_xing_ip8_r
            self.results_on_sep8_x[self.current_time] = self.current_on_sep8_x
            self.results_on_sep8_y[self.current_time] = self.current_on_sep8_y
            self.results_on_sep8_r[self.current_time] = self.current_on_sep8_r

        
        try:
            self.results_lumi_ATLAS_virtual[self.current_time] = self.current_lumi_ATLAS_virtual
            self.results_lumi_CMS_virtual[self.current_time] = self.current_lumi_CMS_virtual
            self.results_lumi_avg_virtual[self.current_time] = np.mean([np.mean(self.current_lumi_ATLAS_virtual), np.mean(self.current_lumi_CMS_virtual)])
            self.results_pu_ATLAS_virtual[self.current_time] = self.current_pu_ATLAS_virtual
            self.results_pu_CMS_virtual[self.current_time] = self.current_pu_CMS_virtual
            self.results_pu_avg_virtual[self.current_time] = np.mean([np.mean(self.current_pu_ATLAS_virtual), np.mean(self.current_pu_CMS_virtual)])
            
            self.results_lumi_ATLAS_virtual_2um[self.current_time] = self.current_lumi_ATLAS_virtual_2um
            self.results_lumi_CMS_virtual_2um[self.current_time] = self.current_lumi_CMS_virtual_2um
            self.results_lumi_avg_virtual_2um[self.current_time] = np.mean([np.mean(self.current_lumi_ATLAS_virtual_2um), np.mean(self.current_lumi_CMS_virtual_2um)])
            self.results_pu_ATLAS_virtual_2um[self.current_time] = self.current_pu_ATLAS_virtual_2um
            self.results_pu_CMS_virtual_2um[self.current_time] = self.current_pu_CMS_virtual_2um
            self.results_pu_avg_virtual_2um[self.current_time] = np.mean([np.mean(self.current_pu_ATLAS_virtual_2um), np.mean(self.current_pu_CMS_virtual_2um)])

        except:
            print("No virtual lumi")
            self.results_lumi_ATLAS_virtual[self.current_time] = 0#self.current_lumi_ATLAS
            self.results_lumi_CMS_virtual[self.current_time] = 0#self.current_lumi_CMS
            self.results_lumi_avg_virtual[self.current_time] = 0#np.mean([np.mean(self.current_lumi_ATLAS), np.mean(self.current_lumi_CMS)])
            self.results_pu_ATLAS_virtual[self.current_time] = 0#self.current_pu_ATLAS
            self.results_pu_CMS_virtual[self.current_time] = 0#self.current_pu_CMS
            self.results_pu_avg_virtual[self.current_time] = 0#np.mean([np.mean(self.current_pu_ATLAS), np.mean(self.current_pu_CMS)])
            self.results_lumi_ATLAS_virtual_2um[self.current_time] = 0#self.current_lumi_ATLAS
            self.results_lumi_CMS_virtual_2um[self.current_time] = 0#self.current_lumi_CMS
            self.results_lumi_avg_virtual_2um[self.current_time] = 0#np.mean([np.mean(self.current_lumi_ATLAS), np.mean(self.current_lumi_CMS)])
            self.results_pu_ATLAS_virtual_2um[self.current_time] = 0#self.current_pu_ATLAS
            self.results_pu_CMS_virtual_2um[self.current_time] = 0#self.current_pu_CMS
            self.results_pu_avg_virtual_2um[self.current_time] = 0#np.mean([np.mean(self.current_pu_ATLAS), np.mean(self.current_pu_CMS)])


    def dump_to_df(self):
        if self.include_lhcb:
            return pd.DataFrame({"betax":self.results_betax,
                      "betax_ip1":self.results_betax_ip1,
                      "betay_ip1":self.results_betay_ip1,
                      "betax_ip5":self.results_betax_ip5,
                      "betay_ip5":self.results_betay_ip5,
                      "betay":self.results_betay,
                      "xing":self.results_xing,
                      "xing_ip1_x": self.results_xing_ip1_x,
                        "xing_ip1_y": self.results_xing_ip1_y,
                        "xing_ip5_x": self.results_xing_ip5_x,
                        "xing_ip5_y": self.results_xing_ip5_y,
                      "on_sep1_x":self.results_on_sep1_x,
                        "on_sep1_y":self.results_on_sep1_y,
                        "on_sep5_x":self.results_on_sep5_x,
                        "on_sep5_y":self.results_on_sep5_y,
                        "emitx_B1":self.results_emitx_B1,
                        "emity_B1":self.results_emity_B1,
                        "emitx_B2":self.results_emitx_B2,
                        "emity_B2":self.results_emity_B2,
                        "emitx_B1_uncoupled":self.results_emitx_B1_uncoupled,
                        "emity_B1_uncoupled":self.results_emity_B1_uncoupled,
                        "emitx_B2_uncoupled":self.results_emitx_B2_uncoupled,
                        "emity_B2_uncoupled":self.results_emity_B2_uncoupled,
                        "bl_s_B1":self.results_bl_s_B1,
                        "bl_s_B2":self.results_bl_s_B2,
                        "inten_B1":self.results_inten_B1,
                        "inten_B2":self.results_inten_B2,
                        "lumi_ATLAS":self.results_lumi_ATLAS,
                        "lumi_CMS":self.results_lumi_CMS,
                        "lumi_avg":self.results_lumi_avg,
                        "pu_ATLAS":self.results_pu_ATLAS,
                        "pu_CMS":self.results_pu_CMS,
                        "pu_avg":self.results_pu_avg,
                        #"longitudinal_luminous_region_ATLAS":self.results_longitudinal_luminous_region_ATLAS,
                        #"longitudinal_luminous_region_CMS":self.results_longitudinal_luminous_region_CMS,
                        #"longitudinal_luminous_region_avg":self.results_longitudinal_luminous_region_avg,
                        "virtual_lumi_ATLAS":self.results_lumi_ATLAS_virtual,
                        "virtual_lumi_CMS":self.results_lumi_CMS_virtual,
                        "virtual_lumi_avg":self.results_lumi_avg_virtual,
                        "virtual_pu_ATLAS":self.results_pu_ATLAS_virtual,
                        "virtual_pu_CMS":self.results_pu_CMS_virtual,
                        "virtual_pu_avg":self.results_pu_avg_virtual,
                        "virtual_lumi_ATLAS_2um":self.results_lumi_ATLAS_virtual_2um,
                        "virtual_lumi_CMS_2um":self.results_lumi_CMS_virtual_2um,
                        "virtual_lumi_avg_2um":self.results_lumi_avg_virtual_2um,
                        "virtual_pu_ATLAS_2um":self.results_pu_ATLAS_virtual_2um,
                        "virtual_pu_CMS_2um":self.results_pu_CMS_virtual_2um,
                        "virtual_pu_avg_2um":self.results_pu_avg_virtual_2um,
                        "lumi_LHCB":self.results_lumi_LHCB,
                        "pu_LHCB":self.results_pu_LHCB,
                        "betax_ip8":self.results_betax_ip8,
                        "betay_ip8":self.results_betay_ip8,
                        "xing_ip8_x":self.results_xing_ip8_x,
                        "xing_ip8_y":self.results_xing_ip8_y,
                        "xing_ip8_r":self.results_xing_ip8_r,
                        "on_sep8_x":self.results_on_sep8_x,
                        "on_sep8_y":self.results_on_sep8_y,
                        "on_sep8_r":self.results_on_sep8_r,
                        "ccvx_ip1":self.results_ccvx_ip1,
                        "ccvy_ip1":self.results_ccvy_ip1,
                        "ccvx_ip5":self.results_ccvx_ip5,
                        "ccvy_ip5":self.results_ccvy_ip5,
                        "time":self.df_init_b1.index})

        else:
            return pd.DataFrame({"betax":self.results_betax,
                      "betax_ip1":self.results_betax_ip1,
                      "betay_ip1":self.results_betay_ip1,
                      "betax_ip5":self.results_betax_ip5,
                      "betay_ip5":self.results_betay_ip5,
                      "betay":self.results_betay,
                      "xing":self.results_xing,
                     "xing_ip1_x": self.results_xing_ip1_x,
                        "xing_ip1_y": self.results_xing_ip1_y,
                        "xing_ip5_x": self.results_xing_ip5_x,
                        "xing_ip5_y": self.results_xing_ip5_y,
                     "on_sep1_x":self.results_on_sep1_x,
                        "on_sep1_y":self.results_on_sep1_y,
                        "on_sep5_x":self.results_on_sep5_x,
                        "on_sep5_y":self.results_on_sep5_y,
                        "emitx_B1":self.results_emitx_B1,
                        "emity_B1":self.results_emity_B1,
                        "emitx_B2":self.results_emitx_B2,
                        "emity_B2":self.results_emity_B2,
                        "emitx_B1_uncoupled":self.results_emitx_B1_uncoupled,
                        "emity_B1_uncoupled":self.results_emity_B1_uncoupled,
                        "emitx_B2_uncoupled":self.results_emitx_B2_uncoupled,
                        "emity_B2_uncoupled":self.results_emity_B2_uncoupled,
                        "bl_s_B1":self.results_bl_s_B1,
                        "bl_s_B2":self.results_bl_s_B2,
                        "inten_B1":self.results_inten_B1,
                        "inten_B2":self.results_inten_B2,
                        "lumi_ATLAS":self.results_lumi_ATLAS,
                        "lumi_CMS":self.results_lumi_CMS,
                        "lumi_avg":self.results_lumi_avg,
                        #"longitudinal_luminous_region_ATLAS":self.results_longitudinal_luminous_region_ATLAS,
                        #"longitudinal_luminous_region_CMS":self.results_longitudinal_luminous_region_CMS,
                        #"longitudinal_luminous_region_avg":self.results_longitudinal_luminous_region_avg,
                        "pu_ATLAS":self.results_pu_ATLAS,
                        "pu_CMS":self.results_pu_CMS,
                        "pu_avg":self.results_pu_avg,
                        "virtual_lumi_ATLAS":self.results_lumi_ATLAS_virtual,
                        "virtual_lumi_CMS":self.results_lumi_CMS_virtual,
                        "virtual_lumi_avg":self.results_lumi_avg_virtual,
                        "virtual_pu_ATLAS":self.results_pu_ATLAS_virtual,
                        "virtual_pu_CMS":self.results_pu_CMS_virtual,
                        "virtual_pu_avg":self.results_pu_avg_virtual,
                        "virtual_lumi_ATLAS_2um":self.results_lumi_ATLAS_virtual_2um,
                        "virtual_lumi_CMS_2um":self.results_lumi_CMS_virtual_2um,
                        "virtual_lumi_avg_2um":self.results_lumi_avg_virtual_2um,
                        "virtual_pu_ATLAS_2um":self.results_pu_ATLAS_virtual_2um,
                        "virtual_pu_CMS_2um":self.results_pu_CMS_virtual_2um,
                        "virtual_pu_avg_2um":self.results_pu_avg_virtual_2um,
                        "ccvx_ip1":self.results_ccvx_ip1,
                        "ccvy_ip1":self.results_ccvy_ip1,
                        "ccvx_ip5":self.results_ccvx_ip5,
                        "ccvy_ip5":self.results_ccvy_ip5,
                        "time":self.df_init_b1.index})


    ########################################################################################################
    # leveling

    def continuous_beta_leveling(self, params, Llevel0, level_on="avg", current_xing="BBLR", betaratio=1, on_sep1=0, on_sep5=0):

        if (self.ATLAS_crossing == "V") and (self.CMS_crossing == "H"):

            betax_ip1 = min(params[0], betaratio*params[0])
            betay_ip1 = max(params[0], betaratio*params[0])
            betax_ip5 = max(params[0], betaratio*params[0])
            betay_ip5 = min(params[0], betaratio*params[0])
        else:
            betax_ip1 = max(params[0], betaratio*params[0])
            betay_ip1 = min(params[0], betaratio*params[0])
            betax_ip5 = min(params[0], betaratio*params[0])
            betay_ip5 = max(params[0], betaratio*params[0])          
                            
        if current_xing=="BBLR":
            myxing = self.cmp_xing_BBLR(max(betax_ip1, betay_ip1))*2.0*1e-6
        else:
            myxing = current_xing

        if self.ATLAS_crossing == "V":
                xing_ip1_y = myxing
                xing_ip1_x = 0
                current_on_sep1_y = 0
                current_on_sep1_x = on_sep1
        else:
                xing_ip1_y = 0
                xing_ip1_x = myxing
                current_on_sep1_x = 0
                current_on_sep1_y = on_sep1
        if self.CMS_crossing == "V":
                xing_ip5_y = myxing
                xing_ip5_x = 0
                current_on_sep5_y = 0
                current_on_sep5_x = on_sep5
        else:
                xing_ip5_y = 0
                xing_ip5_x = myxing
                current_on_sep5_x = 0
                current_on_sep5_y = on_sep5

        current_lumi_ATLAS, current_lumi_CMS = self.compute_CMS_ATLAS_lumi(current_betax_ip1 = betax_ip1, current_betay_ip1 = betay_ip1,
                                                                      current_betax_ip5=betax_ip5, current_betay_ip5=betay_ip5,
                                                                      current_xing_ip1_x=xing_ip1_x, current_xing_ip1_y=xing_ip1_y,
                                                                      current_xing_ip5_x=xing_ip5_x, current_xing_ip5_y=xing_ip5_y,
                                                                      current_on_sep1_y = current_on_sep1_y, current_on_sep1_x = current_on_sep1_x,
                                                                      current_on_sep5_x=current_on_sep5_x, current_on_sep5_y=current_on_sep5_y)

        if level_on == 'avg':
            current_lumi = 0.5*(np.mean(current_lumi_ATLAS) + np.mean(current_lumi_CMS))/1e38
        elif level_on == "ATLAS":
            current_lumi = np.mean(current_lumi_ATLAS)/1e38
        elif level_on == 'CMS':
            current_lumi = np.mean(current_lumi_CMS)/1e38
        return (Llevel0-current_lumi)
        
    def continuous_separation_leveling(self, params, Llevel0, level_on="CMS", verbose=False):
                     
        if self.ATLAS_crossing == "V":
            current_on_sep1_y = 0
            current_on_sep1_x = params[0]
        else:
            current_on_sep1_x = 0
            current_on_sep1_y = params[0]
        if self.CMS_crossing == "V":
            current_on_sep5_y = 0
            current_on_sep5_x = params[0]
        else:
            current_on_sep5_x = 0
            current_on_sep5_y = params[0]

        current_lumi_ATLAS, current_lumi_CMS = self.compute_CMS_ATLAS_lumi(current_on_sep1_y = current_on_sep1_y, current_on_sep1_x = current_on_sep1_x,
                                                                           current_on_sep5_x=current_on_sep5_x, current_on_sep5_y=current_on_sep5_y)
        if level_on == 'ATLAS':
            current_lumi = np.mean(current_lumi_ATLAS)/1e38
        elif level_on == 'CMS':
            current_lumi = np.mean(current_lumi_CMS)/1e38
        
        return (Llevel0-current_lumi)


    
    def continuous_xing_leveling(self, params, Llevel0, level_on="CMS", verbose=False):

        if self.ATLAS_crossing == "V":
            current_xing_ip1_x = 0
            current_xing_ip1_y = params[0]
        else:
            current_xing_ip1_y = 0
            current_xing_ip1_x = params[0]
        if self.CMS_crossing == "V":
            current_xing_ip5_x = 0
            current_xing_ip5_y = params[0]
        else:
            current_xing_ip5_y = 0
            current_xing_ip5_x = params[0]

        current_lumi_ATLAS, current_lumi_CMS = self.compute_CMS_ATLAS_lumi(current_xing_ip1_x = current_xing_ip1_x, current_xing_ip1_y = current_xing_ip1_y,
                                                                           current_xing_ip5_x=current_xing_ip5_x, current_xing_ip5_y=current_xing_ip5_y)
        
        if level_on == 'ATLAS':
            current_lumi = np.mean(current_lumi_ATLAS)/1e38
        elif level_on == 'CMS':
            current_lumi = np.mean(current_lumi_CMS)/1e38
        
        return (Llevel0-current_lumi)

 
    
    def continuous_separation_leveling_ip8(self, params, Llevel0, verbose=False):
                  
        
        x_1 = -params[0]*np.sin(self.phi_ip8)
        y_1 = params[0]*np.cos(self.phi_ip8)
                            
        current_lumi_LHCB = self.Lumi_integrals(self.number_of_bunches_coll_ip8, betax = self.betax_ip8, betay= self.betay_ip8, x=x_1, px=self.xing_ip8_x/2., y=y_1, py=self.xing_ip8_y/2., ccx=0, ccy=0, ccrf=400e6,verbose=False)
        current_lumi = np.mean(current_lumi_LHCB)/1e38

        if verbose:
            print("Leveling IP8 by separation")
            print("current_lumi", current_lumi)
            print("Llevel0", Llevel0)
            print("params[0]", params[0])
            print(self.number_of_bunches_coll_ip8)
            print(self.current_betax_ip8, self.current_betay_ip8)
            print(self.current_xing_ip8_x, self.current_xing_ip8_y, np.rad2deg(self.phi_ip8))
            print("")

        return (Llevel0-current_lumi)      
    

    # leveling methods
        
    def level_ip15_lumi_beta_discrete(self, target=None, min_beta=None, max_beta=None):
        if min_beta is None:
            temp_df_optics = self.df_optics
        else:
            temp_df_optics = self.df_optics[ (self.df_optics.betax >= min_beta) & (self.df_optics.betay >=min_beta)]

        if max_beta is not None:
            temp_df_optics = temp_df_optics[ (temp_df_optics.betax <= max_beta) & (temp_df_optics.betay <= max_beta)]
        
        if self.flag_lumi_simple:
            if (self.ATLAS_crossing=="V") and (self.CMS_crossing=="H"):
                allbeta_Lumi_init_CMS = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                    self.current_emitx_B1, self.current_emity_B1,
                                                                    self.current_emitx_B2, self.current_emity_B2,
                                                                    self.current_bl_s_B1, self.current_bl_s_B2,
                                                                    max(current_betax, current_betay), min(current_betax, current_betay), current_xing*2.0*1e-6, "CMS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                allbeta_Lumi_init_ATLAS = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                        self.current_emitx_B1, self.current_emity_B1,
                                                                        self.current_emitx_B2, self.current_emity_B2,
                                                                        self.current_bl_s_B1, self.current_bl_s_B2,
                                                                        min(current_betax, current_betay), max(current_betax, current_betay), current_xing*2.0*1e-6, "ATLAS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
            else:
                allbeta_Lumi_init_CMS = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                    self.current_emitx_B1, self.current_emity_B1,
                                                                    self.current_emitx_B2, self.current_emity_B2,
                                                                    self.current_bl_s_B1, self.current_bl_s_B2,
                                                                    min(current_betax, current_betay), max(current_betax, current_betay), current_xing*2.0*1e-6, "CMS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                allbeta_Lumi_init_ATLAS = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                        self.current_emitx_B1, self.current_emity_B1,
                                                                        self.current_emitx_B2, self.current_emity_B2,
                                                                        self.current_bl_s_B1, self.current_bl_s_B2,
                                                                        max(current_betax, current_betay), min(current_betax, current_betay), current_xing*2.0*1e-6, "ATLAS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
        else:
            if (self.ATLAS_crossing=="V") and (self.CMS_crossing=="H"):
                allbeta_Lumi_init_CMS = [np.mean(self.Lumi_integrals(self.number_of_bunches_coll_ip15, 
                                                        betax = max(current_betax, current_betay), betay= min(current_betax, current_betay), 
                                                        x=0, px=current_xing*1e-6, y=0, py=0, ccx=self.ccx_ip5, ccy=self.ccy_ip5, ccrf=400e6,verbose=False)) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                allbeta_Lumi_init_ATLAS = [np.mean(self.Lumi_integrals(self.number_of_bunches_coll_ip15, 
                                                        betax = min(current_betax, current_betay), betay= max(current_betax, current_betay), 
                                                        x=0, px=0, y=0, py=current_xing*1e-6, ccx=self.ccx_ip1, ccy=self.ccy_ip1, ccrf=400e6,verbose=False)) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
            else:
                allbeta_Lumi_init_CMS = [np.mean(self.Lumi_integrals(self.number_of_bunches_coll_ip15, 
                                                        betax = min(current_betax, current_betay), betay= max(current_betax, current_betay), 
                                                        x=0, px=current_xing*1e-6, y=0, py=0, ccx=self.ccx_ip5, ccy=self.ccy_ip5, ccrf=400e6,verbose=False)) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                allbeta_Lumi_init_ATLAS = [np.mean(self.Lumi_integrals(self.number_of_bunches_coll_ip15, 
                                                        betax = max(current_betax, current_betay), betay= min(current_betax, current_betay), 
                                                        x=0, px=0, y=0, py=current_xing*1e-6, ccx=self.ccx_ip1, ccy=self.ccy_ip1, ccrf=400e6,verbose=False)) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
        
        if self.level_on_ATLAS_or_CMS == 'avg':
            allbeta_Lumi_init = 0.5*(np.array(allbeta_Lumi_init_ATLAS) + np.array(allbeta_Lumi_init_CMS))
            mytarget = 0.5*(self.level_lumi_ip1 + self.level_lumi_ip5)
        elif self.level_on_ATLAS_or_CMS == 'ATLAS':
            allbeta_Lumi_init = np.array(allbeta_Lumi_init_ATLAS)
            mytarget = self.level_lumi_ip1
        elif self.level_on_ATLAS_or_CMS == 'CMS':
            allbeta_Lumi_init = np.array(allbeta_Lumi_init_CMS)
            mytarget = self.level_lumi_ip5

        if target is None:
            target = mytarget

        beta_idx = np.nanargmin(abs(allbeta_Lumi_init-target))

        if (allbeta_Lumi_init[beta_idx]<target - self.lumi_tol_percentage/100.0*target) and (beta_idx < len(allbeta_Lumi_init)-2):
            beta_idx = beta_idx + 1

        #if allbeta_Lumi_init[beta_idx]>target: #+ self.lumi_tol_percentage/100.0*target:
        #    # go back one beta* step
        #    beta_idx = beta_idx - 1

        #elif allbeta_Lumi_init[beta_idx]<target - self.lumi_tol_percentage/100.0*target and \
        #    beta_idx < len(allbeta_Lumi_init)-1:
        #    # go forward one beta* step
        #    beta_idx = beta_idx + 1

        self.current_betax = temp_df_optics.betax.values[beta_idx]
        self.current_betay = temp_df_optics.betay.values[beta_idx]
        self.current_xing = temp_df_optics.xing.values[beta_idx]*2.0*1e-6


        if min(self.current_betax, self.current_betay) <= min(temp_df_optics.iloc[-1]["betax"], temp_df_optics.iloc[-1]["betay"]):
            self.beta_leveling_ended = True

        # Adjust for ATLAS crossing
        if self.ATLAS_crossing == "V":
            self.current_betax_ip1, self.current_betay_ip1 = sorted([self.current_betax, self.current_betay])
            self.current_xing_ip1_y, self.current_xing_ip1_x = self.current_xing, 0.0
        else:
            self.current_betay_ip1, self.current_betax_ip1 = sorted([self.current_betax, self.current_betay])
            self.current_xing_ip1_x, self.current_xing_ip1_y = self.current_xing, 0.0

        # Adjust for CMS crossing
        if self.CMS_crossing == "H":
            self.current_betax_ip5, self.current_betay_ip5 = sorted([self.current_betax, self.current_betay], reverse=True)
            self.current_xing_ip5_x, self.current_xing_ip5_y = self.current_xing, 0.0
        else:
            self.current_betay_ip5, self.current_betax_ip5 = sorted([self.current_betax, self.current_betay], reverse=True)
            self.current_xing_ip5_y, self.current_xing_ip5_x = self.current_xing, 0.0

        self.beta_idx = beta_idx

        self.current_lumi_ATLAS, self.current_lumi_CMS = self.compute_CMS_ATLAS_lumi()
        #self.current_longitudinal_luminous_region_ATLAS, self.current_longitudinal_luminous_region_CMS = self.compute_CMS_ATLAS_lumi(flag="longitudinal luminous region")

        self.current_pu_ATLAS = self.cmp_pileup(self.current_lumi_ATLAS)
        self.current_pu_CMS = self.cmp_pileup(self.current_lumi_CMS)

        if self.verbose:
            print(f"Time step {self.current_time}: Average lumi between the two experiments is lower than level lumi {target}. Going to matching point: betax* = ", temp_df_optics.betax.values[beta_idx], temp_df_optics.betay.values[beta_idx], "m and xing = ", temp_df_optics.xing.values[beta_idx], "um which results in lumi ", allbeta_Lumi_init[beta_idx], "Hz/m2")


    def level_ip15_lumi_beta_continuous(self, target=None, min_beta=None, max_beta=None):
        if target is None:
            target = self.level_lumi_ip1
        if self.flag_xing_ct:
            current_xing = self.xing_ref * 2.0 * 1e-6
        else:
            current_xing = "BBLR"

        if not self.beta_continuous_leveling_ended:
          bounds = ([1e-6], [np.inf])
          res = least_squares(self.continuous_beta_leveling, [self.initial_beta_guess], args=(target / 1e38, self.level_on_ATLAS_or_CMS, current_xing, self.betaratio, 0, 0), bounds=bounds)
          current_beta_x = res['x'][0]
          current_beta_y = current_beta_x * self.betaratio
        else:
          current_beta_x = self.current_betax
          current_beta_y = self.current_betay

        self.initial_beta_guess = min(current_beta_x, current_beta_y)
        if min_beta is None:
            if current_beta_x <= self.df_optics.iloc[-1]["betax"]:
                current_beta_x = self.df_optics.iloc[-1]["betax"]
                current_beta_y = current_beta_x * self.betaratio
                self.beta_continuous_leveling_ended = True

            if current_beta_y <= self.df_optics.iloc[-1]["betay"]:
                current_beta_y = self.df_optics.iloc[-1]["betay"]
                current_beta_x = current_beta_y * self.betaratio
                self.beta_continuous_leveling_ended = True
        else:
            if current_beta_x <= min_beta:
                current_beta_x = min_beta
                current_beta_y = current_beta_x * self.betaratio
                self.beta_continuous_leveling_ended = True

            if current_beta_y <= min_beta:
                current_beta_y = min_beta
                current_beta_x = current_beta_y * self.betaratio
                self.beta_continuous_leveling_ended= True
        
        if max_beta is not None:
            if current_beta_x >= max_beta:
                current_beta_x = max_beta
                current_beta_y = current_beta_x * self.betaratio
                self.beta_continuous_leveling_ended = True

            if current_beta_y >= max_beta:
                current_beta_y = max_beta
                current_beta_x = current_beta_y * self.betaratio
                self.beta_continuous_leveling_ended= True

        self.current_betax = current_beta_x
        self.current_betay = current_beta_y
        if self.flag_xing_ct:
            self.current_xing = self.xing_ref * 2.0 * 1e-6
        else:
            self.current_xing = self.cmp_xing_BBLR(max(self.current_betax, self.current_betay)) * 2.0 * 1e-6

        # Adjust for ATLAS crossing
        if self.ATLAS_crossing == "V":
            self.current_betax_ip1, self.current_betay_ip1 = sorted([self.current_betax, self.current_betay])
            self.current_xing_ip1_y, self.current_xing_ip1_x = self.current_xing, 0.0
        else:
            self.current_betay_ip1, self.current_betax_ip1 = sorted([self.current_betax, self.current_betay])
            self.current_xing_ip1_x, self.current_xing_ip1_y = self.current_xing, 0.0

        # Adjust for CMS crossing
        if self.CMS_crossing == "H":
            self.current_betax_ip5, self.current_betay_ip5 = sorted([self.current_betax, self.current_betay], reverse=True)
            self.current_xing_ip5_x, self.current_xing_ip5_y = self.current_xing, 0.0
        else:
            self.current_betay_ip5, self.current_betax_ip5 = sorted([self.current_betax, self.current_betay], reverse=True)
            self.current_xing_ip5_y, self.current_xing_ip5_x = self.current_xing, 0.0

        self.current_lumi_ATLAS, self.current_lumi_CMS = self.compute_CMS_ATLAS_lumi()
        # self.current_longitudinal_luminous_region_ATLAS, self.current_longitudinal_luminous_region_CMS = self.compute_CMS_ATLAS_lumi(flag="longitudinal luminous region")

        self.current_pu_ATLAS = self.cmp_pileup(self.current_lumi_ATLAS)
        self.current_pu_CMS = self.cmp_pileup(self.current_lumi_CMS)

        # print(self.current_betax_ip1, self.current_betay_ip1, self.current_betax_ip5, self.current_betay_ip5)
        # print(self.current_lumi_avg, self.current_lumi_ATLAS, self.current_lumi_CMS)

    def level_ip15_virtual_lumi_beta_discrete(self, level_virtual_emit=2, min_beta=None, target=None):

        if min_beta is None:
            temp_df_optics = self.df_optics
        else:
            temp_df_optics = self.df_optics[ (self.df_optics.betax >= min_beta) & (self.df_optics.betay >= min_beta)]

        if self.flag_lumi_simple:
                        
            if (self.ATLAS_crossing=="V") and (self.CMS_crossing=="H"):

                        allbeta_Lumi_init_CMS_virtual = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                        self.current_emitx_B1, self.current_emity_B1,
                                                                        self.current_emitx_B2, self.current_emity_B2,
                                                                        self.current_bl_s_B1, self.current_bl_s_B2,
                                                                        max(current_betax, current_betay), min(current_betax, current_betay), current_xing*2.0*1e-6, "CMS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values,temp_df_optics.betay.values, temp_df_optics.xing.values)]
                        allbeta_Lumi_init_ATLAS_virtual = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                                self.current_emitx_B1, self.current_emity_B1,
                                                                                self.current_emitx_B2, self.current_emity_B2,
                                                                                self.current_bl_s_B1, self.current_bl_s_B2,
                                                                                min(current_betax, current_betay), max(current_betax, current_betay), current_xing*2.0*1e-6, "ATLAS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                        allbeta_Lumi_init_CMS_virtual_2um = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                        self.virtual_lumi_emit, self.virtual_lumi_emit,
                                                                        self.virtual_lumi_emit, self.virtual_lumi_emit,
                                                                        self.current_bl_s_B1, self.current_bl_s_B2,
                                                                        max(current_betax, current_betay), min(current_betax, current_betay), current_xing*2.0*1e-6, "CMS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                        allbeta_Lumi_init_ATLAS_virtual_2um = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                                self.virtual_lumi_emit, self.virtual_lumi_emit,
                                                                                self.virtual_lumi_emit, self.virtual_lumi_emit,
                                                                                self.current_bl_s_B1, self.current_bl_s_B2,
                                                                                min(current_betax, current_betay), max(current_betax, current_betay), current_xing*2.0*1e-6, "ATLAS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
            else:
                        allbeta_Lumi_init_CMS_virtual = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                        self.current_emitx_B1, self.current_emity_B1,
                                                                        self.current_emitx_B2, self.current_emity_B2,
                                                                        self.current_bl_s_B1, self.current_bl_s_B2,
                                                                        min(current_betax, current_betay), max(current_betax, current_betay), current_xing*2.0*1e-6, "CMS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                        allbeta_Lumi_init_ATLAS_virtual = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                                self.current_emitx_B1, self.current_emity_B1,
                                                                                self.current_emitx_B2, self.current_emity_B2,
                                                                                self.current_bl_s_B1, self.current_bl_s_B2,
                                                                                max(current_betax, current_betay), min(current_betax, current_betay), current_xing*2.0*1e-6, "ATLAS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                        allbeta_Lumi_init_CMS_virtual_2um = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                        self.virtual_lumi_emit, self.virtual_lumi_emit,
                                                                        self.virtual_lumi_emit, self.virtual_lumi_emit,
                                                                        self.current_bl_s_B1, self.current_bl_s_B2,
                                                                        min(current_betax, current_betay), max(current_betax, current_betay), current_xing*2.0*1e-6, "CMS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                        allbeta_Lumi_init_ATLAS_virtual_2um = [np.mean(self.Lumi_inst_IP15(self.current_inten_B1, self.current_inten_B2,
                                                                                self.virtual_lumi_emit, self.virtual_lumi_emit,
                                                                                self.virtual_lumi_emit, self.virtual_lumi_emit,
                                                                                self.current_bl_s_B1, self.current_bl_s_B2,
                                                                                max(current_betax, current_betay), min(current_betax, current_betay), current_xing*2.0*1e-6, "ATLAS")) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                                                                                           
                

        else:
                        allbeta_Lumi_init_CMS_virtual = [np.mean(self.Lumi_integrals(self.number_of_bunches_coll_ip15, 
                                                                   betax = max(current_betax, current_betay), betay= min(current_betax, current_betay), 
                                                                   x=0, px=current_xing*1e-6, y=0, py=0, ccx=self.ccx_ip5, ccy=self.ccy_ip5, ccrf=400e6,verbose=False)) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                        allbeta_Lumi_init_ATLAS_virtual = [np.mean(self.Lumi_integrals(self.number_of_bunches_coll_ip15, 
                                                                   betax = min(current_betax, current_betay), betay= max(current_betax, current_betay), 
                                                                   x=0, px=0, y=0, py=current_xing*1e-6, ccx=self.ccx_ip1, ccy=self.ccy_ip1,ccrf=400e6,verbose=False)) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]

                        allbeta_Lumi_init_CMS_virtual_2um = [np.mean(self.Lumi_integrals(self.number_of_bunches_coll_ip15, 
                                                                   betax = max(current_betax, current_betay), betay= min(current_betax, current_betay), 
                                                                   x=0, px=current_xing*1e-6, y=0, py=0, ccx=self.ccx_ip5, ccy=self.ccy_ip5, ccrf=400e6,verbose=False, emit=self.virtual_lumi_emit)) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]
                        allbeta_Lumi_init_ATLAS_virtual_2um = [np.mean(self.Lumi_integrals(self.number_of_bunches_coll_ip15, 
                                                                   betax = min(current_betax, current_betay), betay= max(current_betax, current_betay), 
                                                                   x=0, px=0, y=0, py=current_xing*1e-6, ccx=self.ccx_ip1, ccy=self.ccy_ip1, ccrf=400e6,verbose=False, emit=self.virtual_lumi_emit)) for current_betax, current_betay, current_xing in zip(temp_df_optics.betax.values, temp_df_optics.betay.values, temp_df_optics.xing.values)]


        allbeta_Lumi_init_virtual = np.array(allbeta_Lumi_init_CMS_virtual)
        allbeta_Lumi_init_virtual_2um = np.array(allbeta_Lumi_init_CMS_virtual_2um)

        if target is None:
            self.level_by_pu(how="virtual")
            target = self.level_virtual_lumi

        beta_idx = np.nanargmin(abs(allbeta_Lumi_init_virtual-target))
        beta_idx_2um = np.nanargmin(abs(allbeta_Lumi_init_virtual_2um-target))

        if level_virtual_emit == 2:
            if allbeta_Lumi_init_virtual_2um[beta_idx_2um]>target + self.virtual_lumi_tol_percentage/100.0*target:
                # go back one beta* step
                beta_idx_2um = beta_idx_2um - 1
            beta_idx = beta_idx_2um
        else:
            if allbeta_Lumi_init_virtual[beta_idx]>target + self.virtual_lumi_tol_percentage/100.0*target:
                beta_idx = beta_idx - 1

        self.current_betax = temp_df_optics.betax.values[beta_idx]
        self.current_betay = temp_df_optics.betay.values[beta_idx]
        self.current_xing = temp_df_optics.xing.values[beta_idx]*2.0*1e-6

        if min(self.current_betax, self.current_betay) <= min(temp_df_optics.iloc[-1]["betax"], temp_df_optics.iloc[-1]["betay"]):
                self.beta_leveling_ended = True
        

        # Adjust for ATLAS crossing
        if self.ATLAS_crossing == "V":
            self.current_betax_ip1, self.current_betay_ip1 = sorted([self.current_betax, self.current_betay])
            self.current_xing_ip1_y, self.current_xing_ip1_x = self.current_xing, 0.0
        else:
            self.current_betay_ip1, self.current_betax_ip1 = sorted([self.current_betax, self.current_betay])
            self.current_xing_ip1_x, self.current_xing_ip1_y = self.current_xing, 0.0

        # Adjust for CMS crossing
        if self.CMS_crossing == "H":
            self.current_betax_ip5, self.current_betay_ip5 = sorted([self.current_betax, self.current_betay], reverse=True)
            self.current_xing_ip5_x, self.current_xing_ip5_y = self.current_xing, 0.0
        else:
            self.current_betay_ip5, self.current_betax_ip5 = sorted([self.current_betax, self.current_betay], reverse=True)
            self.current_xing_ip5_y, self.current_xing_ip5_x = self.current_xing, 0.0

        self.beta_idx = beta_idx

        self.current_lumi_ATLAS_virtual_2um, self.current_lumi_CMS_virtual_2um = self.compute_CMS_ATLAS_lumi(current_emitx_B1=self.virtual_lumi_emit, current_emity_B1=self.virtual_lumi_emit, current_emitx_B2=self.virtual_lumi_emit, current_emity_B2=self.virtual_lumi_emit, current_on_sep1_x=0, current_on_sep1_y=0, current_on_sep5_x=0, current_on_sep5_y=0)
        self.current_lumi_ATLAS_virtual, self.current_lumi_CMS_virtual = self.compute_CMS_ATLAS_lumi(current_on_sep1_x=0, current_on_sep1_y=0, current_on_sep5_x=0, current_on_sep5_y=0)

        self.current_pu_ATLAS_virtual = self.cmp_pileup(self.current_lumi_ATLAS_virtual)
        self.current_pu_CMS_virtual = self.cmp_pileup(self.current_lumi_CMS_virtual)
        self.current_pu_ATLAS_virtual_2um = self.cmp_pileup(self.current_lumi_ATLAS_virtual_2um)
        self.current_pu_CMS_virtual_2um = self.cmp_pileup(self.current_lumi_CMS_virtual_2um)

        if self.verbose:
            print(f"Time step {self.current_time_step}: Average VIRTUAL lumi between the two experiments is lower than level lumi {self.level_virtual_lumi}. Going to matching point: betax* = ", self.df_optics.betax.values[beta_idx], self.df_optics.betay.values[beta_idx], "m and xing = ", self.df_optics.xing.values[beta_idx], "um which results in lumi ", allbeta_Lumi_init_virtual[beta_idx], "Hz/m2")

        self.current_lumi_ATLAS, self.current_lumi_CMS = self.compute_CMS_ATLAS_lumi()


    def level_ip15_lumi_separation(self, experiment="CMS", target=None, min_sep=0, max_sep=None):
        if target is None:
            if experiment == "CMS":
                target = self.level_lumi_ip5
            elif experiment == "ATLAS":
                target = self.level_lumi_ip1
        if experiment=="CMS":
            sigma_y = np.sqrt(self.current_emitx_B1[0]/self.gamma * np.min((self.current_betay_ip5, self.current_betax_ip5)))
            res = least_squares(self.continuous_separation_leveling, [0.5*sigma_y], args=(target/1e38, "CMS", False))
            sep_ip5 = res['x'][0]
            if sep_ip5<=min_sep:
                self.separation_leveling_ended_ip5  = True
                sep_ip5 = 0
            
            if self.CMS_crossing == "H":
                self.current_on_sep5_x = 0.0
                self.current_on_sep5_y = sep_ip5
            else:
                self.current_on_sep5_x = sep_ip5
                self.current_on_sep5_y = 0.0

        elif experiment=="ATLAS":
            sigma_x = np.sqrt(self.current_emitx_B1[0]/self.gamma* np.min((self.current_betay_ip1, self.current_betax_ip1)))
            res = least_squares(self.continuous_separation_leveling, [0.5*sigma_x], args=(target/1e38, "ATLAS", False))
            sep_ip1 = res['x'][0]
            if sep_ip1<=min_sep:
                self.separation_leveling_ended_ip1  = True
                sep_ip1 =0
            if self.ATLAS_crossing == "V":
                self.current_on_sep1_x = sep_ip1
                self.current_on_sep1_y = 0.0
            else:
                self.current_on_sep1_x = 0.0
                self.current_on_sep1_y = sep_ip1

        self.current_lumi_ATLAS, self.current_lumi_CMS = self.compute_CMS_ATLAS_lumi()
        return res
    

    def level_ip15_lumi_xing(self, experiment="CMS", target=None, min_xing=None, max_xing=None):
        if target is None:
            if experiment == "CMS":
                target = self.level_lumi_ip5
            elif experiment == "ATLAS":
                target = self.level_lumi_ip1
        if experiment == "CMS":
            res = least_squares(self.continuous_xing_leveling, [self.current_xing], args=(target/1e38, "CMS", False))
            myxing = res['x'][0]

            if min_xing is not None:
                if myxing < min_xing:
                    myxing = min_xing
                    self.xing_leveling_ended_ip5  = True
            if max_xing is not None:
                if myxing > max_xing:
                    myxing = max_xing
                    self.xing_leveling_ended_ip5  = True

            if self.CMS_crossing == "H":
                self.current_xing_ip5_y = 0.0
                self.current_xing_ip5_x = myxing
            else:
                self.current_xing_ip5_x = 0.0
                self.current_xing_ip5_y = myxing


        elif experiment == "ATLAS":
            res = least_squares(self.continuous_xing_leveling, [self.current_xing], args=(target/1e38, "ATLAS", False))
            myxing = res['x'][0]
            if min_xing is not None:
                if myxing < min_xing:
                    myxing = min_xing
                    self.xing_leveling_ended_ip1  = True
            if max_xing is not None:
                if myxing > max_xing:
                    myxing = max_xing
                    self.xing_leveling_ended_ip1  = True
            if self.ATLAS_crossing == "H":
                self.current_xing_ip1_y = 0.0
                self.current_xing_ip1_x = myxing
            else:
                self.current_xing_ip1_x = 0.0
                self.current_xing_ip1_y = myxing

        self.current_lumi_ATLAS, self.current_lumi_CMS = self.compute_CMS_ATLAS_lumi()
        return res

    def level_ip8_by_orthogonal_separation(self, target=None):
        if target is None:
            target = self.level_lumi_ip8
        sigma_x = 0.5e-6#np.sqrt(myLumiModel.current_emitx_B1[0]/myLumiModel.gamma * myLumiModel.current_betax_ip8)
        res  = least_squares(self.continuous_separation_leveling_ip8, [0.5*sigma_x], args=(target/1e38, False))
        self.current_on_sep8_x= - res['x'][0]*np.sin(self.phi_ip8)
        self.current_on_sep8_y= res['x'][0]*np.cos(self.phi_ip8)
        self.current_on_sep8_r= np.sqrt(self.current_on_sep8_x**2 + self.current_on_sep8_y**2)
        self.current_lumi_LHCB = self.Lumi_integrals(self.number_of_bunches_coll_ip8,
                                            betax = self.current_betax_ip8, betay= self.current_betay_ip8, 
                                            x=self.current_on_sep8_x, px=self.current_xing_ip8_x/2., y=self.current_on_sep8_y, py=self.current_xing_ip8_y/2., verbose=False)
        self.current_pu_LHCB = self.cmp_pileup(self.current_lumi_LHCB, how="ip8")
        return res
    

    def set_specific_beta(self, beta_idx):
        temp_df_optics = self.df_optics
        self.current_betax = temp_df_optics.betax.values[beta_idx]
        self.current_betay = temp_df_optics.betay.values[beta_idx]
        self.current_xing = temp_df_optics.xing.values[beta_idx]*2.0*1e-6

        if min(self.current_betax, self.current_betay) <= min(temp_df_optics.iloc[-1]["betax"], temp_df_optics.iloc[-1]["betay"]):
                self.beta_leveling_ended = True
        

        # Adjust for ATLAS crossing
        if self.ATLAS_crossing == "V":
            self.current_betax_ip1, self.current_betay_ip1 = sorted([self.current_betax, self.current_betay])
            self.current_xing_ip1_y, self.current_xing_ip1_x = self.current_xing, 0.0
        else:
            self.current_betay_ip1, self.current_betax_ip1 = sorted([self.current_betax, self.current_betay])
            self.current_xing_ip1_x, self.current_xing_ip1_y = self.current_xing, 0.0

        # Adjust for CMS crossing
        if self.CMS_crossing == "H":
            self.current_betax_ip5, self.current_betay_ip5 = sorted([self.current_betax, self.current_betay], reverse=True)
            self.current_xing_ip5_x, self.current_xing_ip5_y = self.current_xing, 0.0
        else:
            self.current_betay_ip5, self.current_betax_ip5 = sorted([self.current_betax, self.current_betay], reverse=True)
            self.current_xing_ip5_y, self.current_xing_ip5_x = self.current_xing, 0.0

        self.beta_idx = beta_idx

        self.current_lumi_ATLAS_virtual_2um, self.current_lumi_CMS_virtual_2um = self.compute_CMS_ATLAS_lumi(current_emitx_B1=self.virtual_lumi_emit, current_emity_B1=self.virtual_lumi_emit, current_emitx_B2=self.virtual_lumi_emit, current_emity_B2=self.virtual_lumi_emit, current_on_sep1_x=0, current_on_sep1_y=0, current_on_sep5_x=0, current_on_sep5_y=0)
        self.current_lumi_ATLAS_virtual, self.current_lumi_CMS_virtual = self.compute_CMS_ATLAS_lumi(current_on_sep1_x=0, current_on_sep1_y=0, current_on_sep5_x=0, current_on_sep5_y=0)

        self.current_pu_ATLAS_virtual = self.cmp_pileup(self.current_lumi_ATLAS_virtual)
        self.current_pu_CMS_virtual = self.cmp_pileup(self.current_lumi_CMS_virtual)
        self.current_pu_ATLAS_virtual_2um = self.cmp_pileup(self.current_lumi_ATLAS_virtual_2um)
        self.current_pu_CMS_virtual_2um = self.cmp_pileup(self.current_lumi_CMS_virtual_2um)

        if self.verbose:
            print(f"Time step: Average VIRTUAL lumi between the two experiments is lower than level lumi {self.level_virtual_lumi}. Going to matching point: betax* = ", self.df_optics.betax.values[beta_idx], self.df_optics.betay.values[beta_idx], "m and xing = ", self.df_optics.xing.values[beta_idx], "um which results in lumi ")

        self.current_lumi_ATLAS, self.current_lumi_CMS = self.compute_CMS_ATLAS_lumi()

    # %%

