# %%
import argparse
import sys
from pathlib import Path
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an HL-LHC luminosity estimate from a YAML configuration."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="YAML configuration file, for example config_2041.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: outputs/<config filename without extension>)",
    )
    return parser.parse_args()


args = parse_args()
config_path = args.config.expanduser().resolve()
if not config_path.is_file():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

output_dir = (
    args.output_dir.expanduser().resolve()
    if args.output_dir
    else Path(__file__).resolve().parent / "outputs" / config_path.stem
)
output_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(config_path, output_dir / config_path.name)

#sys.path.append("/home/HPC/skostogl/workspace_Feb24/HLLHC_lumi_scan/master_jobs/lumi-followup/ibs_module/luminosity_model")
#sys.path.append("/home/HPC/skostogl/workspace_Feb24/HLLHC_lumi_scan/master_jobs/run_lumi_model/scripts")
#sys.path.append("scripts")
#sys.path.append("/afs/cern.ch/work/s/skostogl/private/workspaces_2025/HLLHC/lumi_estimations/Jan2025/master_jobs/run_lumi_model")
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
import fillingpatterns as fp   
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.dates as mdates
from scipy.constants import c as clight
from scipy.optimize import least_squares
import lumi_integrals
from importlib import reload
from datetime import datetime
import tree_maker
import time


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
import lumi_class
reload(lumi_class)

import useful_tools
reload(useful_tools)


# %%

myLumiModel = lumi_class.LumiModel(config_path=str(config_path))

#%%
current_utc_time = datetime.utcnow()
time_range = pd.date_range(start=pd.Timestamp(current_utc_time), periods=myLumiModel.time_periods, freq=f'{myLumiModel.time_step_minutes}min')

df_fake = pd.DataFrame(index=time_range)
df_fake['LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V'] = [np.array([myLumiModel.init_normalized_emittance]) for i in range(len(df_fake))]
df_fake['LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_H'] = [np.array([myLumiModel.init_normalized_emittance]) for i in range(len(df_fake))]
df_fake['LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_V'] = [np.array([myLumiModel.init_normalized_emittance]) for i in range(len(df_fake))]
df_fake['LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_H'] = [np.array([myLumiModel.init_normalized_emittance]) for i in range(len(df_fake))]
df_fake['LHC.BQM.B1:BUNCH_LENGTH_MEAN'] = myLumiModel.init_bunch_length_s
df_fake['LHC.BQM.B2:BUNCH_LENGTH_MEAN'] = myLumiModel.init_bunch_length_s
df_fake['LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY'] = [np.array([myLumiModel.init_intensity]) for i in range(len(df_fake))]
df_fake['LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY'] = [np.array([myLumiModel.init_intensity]) for i in range(len(df_fake))]
df_fake["LHC.RUNCONFIG:IP1-XING-V-MURAD"] = myLumiModel.xing_ref
df_fake["LHC.RUNCONFIG:IP5-XING-H-MURAD"] = myLumiModel.xing_ref
df_fake["HX:BETASTAR_IP1"] = 120.
df_fake["HX:BETASTAR_IP5"] = 120.

#%%
myLumiModel.initialize(df_fake, bbb_or_avg="avg")
# %%
verbose = False
myLumiModel.verbose = verbose

include_coupling = True
include_extraBU  = True
include_CC_noise = True

flag_change_lookup_at_15cm = True

# %%
if myLumiModel.level_lumi_ip1<2.5e38:
    flag_cryo_step = False
    flag_crab_ramp = False
else:
    flag_cryo_step = True

    flag_crab_ramp = True

level_lumi_final = myLumiModel.level_lumi_ip1
if level_lumi_final<=4e+38:
  flag_crab_ramp = False


# cryo step and cc ramp
if flag_cryo_step:
    
    cryo_time_s = np.arange(0, myLumiModel.steps)*myLumiModel.dt_step
    cryo_lumi = np.array([level_lumi_final for i in cryo_time_s])

    x = np.array([10*60, 20*60])
    y = np.array([2.5e38, level_lumi_final])

    slope, intercept = np.polyfit(x, y, 1)
    def linear_function(x):
        return slope * x + intercept

    # Example usage
    for counter, mytime in enumerate(cryo_time_s):
        if mytime<=10*60:
            cryo_lumi[counter] = 2.5e38
        else:
            if linear_function(mytime)<=level_lumi_final:
                cryo_lumi[counter] = linear_function(mytime)
            else:
                continue
    fig, ax = plt.subplots()
    plt.plot(cryo_time_s/60.0, cryo_lumi/1e38)
    plt.xlim(0,30)
    plt.xlabel("Time [min]")
    plt.ylabel("Luminosity [$10^{34}$ Hz/cm$^2$]")
    plt.grid()
else:
    cryo_time_s = np.arange(0, myLumiModel.steps)*myLumiModel.dt_step
    cryo_lumi = np.array([myLumiModel.level_lumi_ip1 for i in cryo_time_s])

if flag_crab_ramp and myLumiModel.include_cc:
    target = myLumiModel.ccx_ip1
    crab_time_s = np.arange(0, myLumiModel.steps)*myLumiModel.dt_step
    crab_angle = np.array([target for i in cryo_time_s])
    x = np.array([0, 20*60])
    y = np.array([0, target])

    slope, intercept = np.polyfit(x, y, 1)
    def linear_function(x):
        return slope * x + intercept

    # Example usage
    for counter, mytime in enumerate(cryo_time_s):
        if linear_function(mytime)>=target:
            crab_angle[counter] = linear_function(mytime)
        else:
            continue

    fig, ax = plt.subplots()
    plt.plot(crab_time_s/60.0, crab_angle*1e6)
    plt.xlim(0,30)
    plt.xlabel("Time [min]")
    plt.ylabel("Half Crab angle [urad]")
    plt.grid()
else:
    crab_time_s = np.arange(0, myLumiModel.steps)*myLumiModel.dt_step
    crab_angle = np.array([0 for i in cryo_time_s])


#%%
start_time = time.time()

for current_time_step in range(0, myLumiModel.steps):

    print(current_time_step, myLumiModel.steps)
    myLumiModel.current_time = current_time_step

    myLumiModel.leveling_ended = False

    if flag_cryo_step:
        myLumiModel.level_lumi_ip1 = cryo_lumi[current_time_step]
        myLumiModel.level_lumi_ip5 = cryo_lumi[current_time_step]
        if myLumiModel.level_lumi_ip1 < level_lumi_final:
            myLumiModel.level_by_pu(how="ip1")
        if myLumiModel.level_lumi_ip5 < level_lumi_final:
            myLumiModel.level_by_pu(how="ip5")

    if myLumiModel.include_cc:
        if (myLumiModel.ATLAS_crossing == "V") and (myLumiModel.CMS_crossing == "H"):
            myLumiModel.ccx_ip1= 0
            myLumiModel.ccy_ip1= crab_angle[current_time_step]
            myLumiModel.ccx_ip5 = crab_angle[current_time_step]
            myLumiModel.ccy_ip5 = 0
        else:
            myLumiModel.ccx_ip1= crab_angle[current_time_step]
            myLumiModel.ccy_ip1= 0
            myLumiModel.ccx_ip5 = 0
            myLumiModel.ccy_ip5 = crab_angle[current_time_step]


    if myLumiModel.current_time == 0:
        myLumiModel.initialize_t0(bunch_family="coll")
        
        #### new feature to simulate blowup observed in operation
        if myLumiModel.use_emittance_data_x == True:
                    print("H Emit from fitted data")
                    bunch_family="coll"
                    time_in_hours0 = (myLumiModel.df_init_b1.index - myLumiModel.df_init_b1.index.min()).total_seconds()/3600.0
                    intercept = myLumiModel.init_normalized_emittance
                    emit_evol_b1h = myLumiModel.ex_extra * time_in_hours0 + intercept
                    emit_evol_b2h = myLumiModel.ex_extra * time_in_hours0 + intercept

                    myLumiModel.ex_extra = 0.0
                    myLumiModel.ex_extra_B1 = 0.0
                    myLumiModel.ex_extra_B2 = 0.0

                    myLumiModel.df_init_b1[f"eh_interp_{bunch_family}"] = [np.array([i]) for i in emit_evol_b1h]
                    myLumiModel.df_init_b2[f"eh_interp_{bunch_family}"] = [np.array([i]) for i in emit_evol_b2h]
                
        if myLumiModel.use_emittance_data_y == True:
                    print("V Emit from fitted data")
                    bunch_family="coll"
                    time_in_hours0 = (myLumiModel.df_init_b1.index - myLumiModel.df_init_b1.index.min()).total_seconds()/3600.0
                    intercept = myLumiModel.init_normalized_emittance
                    emit_evol_b1v = myLumiModel.ey_extra * time_in_hours0 + intercept
                    emit_evol_b2v = myLumiModel.ey_extra * time_in_hours0 + intercept
                    
                    myLumiModel.ey_extra = 0.0
                    myLumiModel.ey_extra_B1 = 0.0
                    myLumiModel.ey_extra_B2 = 0.0

                    myLumiModel.df_init_b1[f"ev_interp_{bunch_family}"] = [np.array([i]) for i in emit_evol_b1v]
                    myLumiModel.df_init_b2[f"ev_interp_{bunch_family}"] = [np.array([i]) for i in emit_evol_b2v]


    # store emittances from previous step before applying any effect
    ex0_B1_coupled = myLumiModel.current_emitx_B1
    ey0_B1_coupled = myLumiModel.current_emity_B1
    ex0_B2_coupled = myLumiModel.current_emitx_B2
    ey0_B2_coupled = myLumiModel.current_emity_B2

    ex0_B1_uncoupled = myLumiModel.current_emitx_B1_uncoupled
    ey0_B1_uncoupled = myLumiModel.current_emity_B1_uncoupled
    ex0_B2_uncoupled = myLumiModel.current_emitx_B2_uncoupled
    ey0_B2_uncoupled = myLumiModel.current_emity_B2_uncoupled



    if myLumiModel.use_emittance_data_x:

            myLumiModel.use_data(bunch_family="coll", current_time_step=myLumiModel.current_time, use_emittance_x = True, use_intensity = False, use_bl = True)
            myLumiModel.ex_extra = 0.0
            myLumiModel.ex_extra_B1 = 0.0
            myLumiModel.ex_extra_B2 = 0.0

    if myLumiModel.use_emittance_data_y:
                   
            myLumiModel.use_data(bunch_family="coll", current_time_step=myLumiModel.current_time, use_emittance_y = True, use_intensity = False, use_bl = True)
            myLumiModel.ey_extra = 0.0
            myLumiModel.ey_extra_B1 = 0.0
            myLumiModel.ey_extra_B2 = 0.0

    myLumiModel.IBS_model(emitx_B1_for_growth_rates=ex0_B1_coupled, emitx_B1_apply_IBS=ex0_B1_uncoupled,
                                    emity_B1_for_growth_rates=ey0_B1_coupled, emity_B1_apply_IBS=ey0_B1_uncoupled,
                                    emitx_B2_for_growth_rates=ex0_B2_coupled, emitx_B2_apply_IBS=ex0_B2_uncoupled,
                                    emity_B2_for_growth_rates=ey0_B2_coupled, emity_B2_apply_IBS=ey0_B2_uncoupled)

    myLumiModel.SR_model(emity_B1_apply_SR=ey0_B1_uncoupled, emity_B2_apply_SR=ey0_B2_uncoupled)


    myLumiModel.Elastic_scattering_model()


    if include_coupling:
            myLumiModel.Coupling_model(emitx_B1=ex0_B1_uncoupled, emity_B1=ey0_B1_uncoupled,
                                            emitx_B2=ex0_B2_uncoupled, emity_B2=ey0_B2_uncoupled)

    if not myLumiModel.use_emittance_data_x:
            myLumiModel.current_emitx_B1_uncoupled = myLumiModel.current_emitx_B1_IBS + myLumiModel.dexdt1 * myLumiModel.dt_step
            myLumiModel.current_emitx_B2_uncoupled = myLumiModel.current_emitx_B2_IBS + myLumiModel.dexdt2 * myLumiModel.dt_step

            myLumiModel.current_emitx_B1 = myLumiModel.current_emitx_B1_uncoupled - myLumiModel.dexy1 * myLumiModel.Cfactor
            myLumiModel.current_emitx_B2 = myLumiModel.current_emitx_B2_uncoupled - myLumiModel.dexy2 * myLumiModel.Cfactor


    if not myLumiModel.use_emittance_data_y:
            myLumiModel.current_emity_B1_uncoupled = myLumiModel.current_emity_B1_IBS_SR + myLumiModel.deydt1 * myLumiModel.dt_step
            myLumiModel.current_emity_B2_uncoupled = myLumiModel.current_emity_B2_IBS_SR + myLumiModel.deydt2 * myLumiModel.dt_step

            myLumiModel.current_emity_B1 = myLumiModel.current_emity_B1_uncoupled + myLumiModel.dexy1 * myLumiModel.Cfactor
            myLumiModel.current_emity_B2 = myLumiModel.current_emity_B2_uncoupled + myLumiModel.dexy2 * myLumiModel.Cfactor
                 

    myLumiModel.current_bl_s_B1 = myLumiModel.current_bl_s_B1_IBS
    myLumiModel.current_bl_s_B2 = myLumiModel.current_bl_s_B2_IBS

    if include_extraBU and (not myLumiModel.use_emittance_data_x) and (not myLumiModel.use_emittance_data_y):
            myLumiModel.ExtraBU_model()

    if include_CC_noise and myLumiModel.include_cc:
            myLumiModel.CCnoise_model()

    myLumiModel.Boff_losses_model()

    if myLumiModel.ATLAS_crossing == "V":
        myLumiModel.current_betax_ip1, myLumiModel.current_betay_ip1 = sorted([myLumiModel.current_betax, myLumiModel.current_betay])
        myLumiModel.current_xing_ip1_y, myLumiModel.current_xing_ip1_x = myLumiModel.current_xing, 0.0
    else:
        myLumiModel.current_betay_ip1, myLumiModel.current_betax_ip1 = sorted([myLumiModel.current_betax, myLumiModel.current_betay])
        myLumiModel.current_xing_ip1_x, myLumiModel.current_xing_ip1_y = myLumiModel.current_xing, 0.0

    if myLumiModel.CMS_crossing == "H":
        myLumiModel.current_betax_ip5, myLumiModel.current_betay_ip5 = sorted([myLumiModel.current_betax, myLumiModel.current_betay], reverse=True)
        myLumiModel.current_xing_ip5_x, myLumiModel.current_xing_ip5_y = myLumiModel.current_xing, 0.0
    else:
        myLumiModel.current_betay_ip5, myLumiModel.current_betax_ip5 = sorted([myLumiModel.current_betax, myLumiModel.current_betay], reverse=True)
        myLumiModel.current_xing_ip5_y, myLumiModel.current_xing_ip5_x = myLumiModel.current_xing, 0.0

    myLumiModel.current_lumi_ATLAS, myLumiModel.current_lumi_CMS = myLumiModel.compute_CMS_ATLAS_lumi()

    if myLumiModel.include_lhcb:
        myLumiModel.current_lumi_LHCB = myLumiModel.Lumi_integrals(myLumiModel.number_of_bunches_coll_ip8,
                                                                   betax=myLumiModel.current_betax_ip8,
                                                                   betay=myLumiModel.current_betay_ip8,
                                                                   x=myLumiModel.current_on_sep8_x,
                                                                   px=myLumiModel.current_xing_ip8_x / 2.,
                                                                   y=myLumiModel.current_on_sep8_y,
                                                                   py=myLumiModel.current_xing_ip8_y / 2.,
                                                                   verbose=False)

    if myLumiModel.level_on_ATLAS_or_CMS == 'avg':
        current_lumi = 0.5 * (np.mean(myLumiModel.current_lumi_ATLAS) + np.mean(myLumiModel.current_lumi_CMS))
        target_lumi = 0.5 * (myLumiModel.level_lumi_ip1 + myLumiModel.level_lumi_ip5)
    elif myLumiModel.level_on_ATLAS_or_CMS == "ATLAS":
        current_lumi = np.mean(myLumiModel.current_lumi_ATLAS)
        target_lumi = myLumiModel.level_lumi_ip1
    elif myLumiModel.level_on_ATLAS_or_CMS == 'CMS':
        current_lumi = np.mean(myLumiModel.current_lumi_CMS)
        target_lumi = myLumiModel.level_lumi_ip5

    if (current_lumi<target_lumi - myLumiModel.lumi_tol_percentage/100.0*target_lumi) and (not myLumiModel.beta_continuous_leveling_ended):
        myLumiModel.level_ip15_lumi_beta_continuous()
        myLumiModel.level_ip15_lumi_separation(experiment="ATLAS")
    elif myLumiModel.beta_continuous_leveling_ended:
        myLumiModel.current_on_sep1_x, myLumiModel.current_on_sep1_y = 0, 0
        myLumiModel.current_on_sep5_x, myLumiModel.current_on_sep5_y = 0, 0
        current_lumi_ATLAS, current_lumi_CMS = myLumiModel.compute_CMS_ATLAS_lumi()
        if np.mean(current_lumi_ATLAS)>myLumiModel.level_lumi_ip1:
            myLumiModel.level_ip15_lumi_separation(experiment="ATLAS")

    if myLumiModel.include_lhcb:
        if (myLumiModel.current_lumi_LHCB > myLumiModel.level_lumi_ip8) or (
                myLumiModel.current_lumi_LHCB < myLumiModel.level_lumi_ip8 - myLumiModel.lumi_tol_percentage_ip8 / 100.0 * myLumiModel.level_lumi_ip8):
            myLumiModel.level_ip8_by_orthogonal_separation()


    myLumiModel.current_lumi_ATLAS , myLumiModel.current_lumi_CMS = myLumiModel.compute_CMS_ATLAS_lumi()
    myLumiModel.current_pu_ATLAS = myLumiModel.cmp_pileup(myLumiModel.current_lumi_ATLAS)
    myLumiModel.current_pu_CMS = myLumiModel.cmp_pileup(myLumiModel.current_lumi_CMS)

    if myLumiModel.include_lhcb:
        myLumiModel.current_lumi_LHCB = myLumiModel.Lumi_integrals(myLumiModel.number_of_bunches_coll_ip8,
                                        betax = myLumiModel.current_betax_ip8, betay= myLumiModel.current_betay_ip8,
                                        x=myLumiModel.current_on_sep8_x, px=myLumiModel.current_xing_ip8_x/2., y=myLumiModel.current_on_sep8_y, py=myLumiModel.current_xing_ip8_y/2., verbose=False)
        myLumiModel.current_pu_LHCB = myLumiModel.cmp_pileup(myLumiModel.current_lumi_LHCB, how="ip8")

    if ((myLumiModel.current_betax  == 0.15) or (myLumiModel.current_betay  == 0.15)) and (flag_change_lookup_at_15cm):
            myLumiModel.current_lookup_table = myLumiModel.lookup_table[0.15]

    myLumiModel.save_data()

end_time = time.time()
print(f"Finished in {end_time-start_time}s")
results = myLumiModel.dump_to_df()
print("Finished!")

results.to_parquet(output_dir / "results.parquet")

# %%
myresults, fig, ax, opt_fill_length, level_time_ATLAS, level_time_CMS =useful_tools.plot_lumi_profiles(results, myLumiModel=myLumiModel)
fig.savefig(output_dir / "1.png")
myresults, fig, ax = useful_tools.plot_ip15_separation(results, myLumiModel, emit_0=myLumiModel.init_normalized_emittance, flag_virtual=False)
fig.savefig(output_dir / "2.png")
#%%
results, res_density, res_pu_density = useful_tools.cmp_longitudinal_luminous_region(results, myLumiModel, verbose=False)
fig, ax = useful_tools.plot_longitudinal_luminous_region(results, myLumiModel, emit_0=myLumiModel.init_normalized_emittance)
fig.savefig(output_dir / "3.png")

# transverse (horizontal + vertical) luminous region, same idea as the
# longitudinal one above but scanning the collision offset in x/y instead of z
#results, res_density_trans, res_pu_density_trans = useful_tools.cmp_transverse_luminous_region(results, myLumiModel, direction="both", verbose=False)
#fig, ax = useful_tools.plot_transverse_luminous_region(results, myLumiModel, direction="both")
#fig.savefig(output_dir / "3b.png")

if myLumiModel.include_lhcb:
  fig, ax = useful_tools.plot_ip8(results, myLumiModel, level_time_ip15=level_time_ATLAS, opt_fill_length=opt_fill_length, emit_0=2.0)
  plt.sca(ax[0,0])
  plt.ylim(0, 74)
  fig.savefig(output_dir / "4.png")
  fig, ax = useful_tools.plot_ip8_longitudinal_luminous_region(results, opt_fill_length=opt_fill_length)
  fig.savefig(output_dir / "5.png")
  #fig, ax = useful_tools.plot_ip8_transverse_luminous_region(results, opt_fill_length=opt_fill_length, direction="both")
  #fig.savefig(output_dir / "5b.png")

# %%
results.to_parquet(output_dir / "results.parquet")
print(f"Results saved in {output_dir}")
# %%
