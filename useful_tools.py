#from scipy.integrate import cumtrapz
try:
    from scipy.integrate import cumtrapz
except ImportError:
    # scipy >= 1.14 renamed cumtrapz to cumulative_trapezoid
    from scipy.integrate import cumulative_trapezoid as cumtrapz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from scipy.constants import c as clight
import lumi_integrals

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



def plot_lumi_profiles(results, myLumiModel=None, config="config.yaml"):
    if myLumiModel is None:
        import yaml
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
        op_days = config["performance"]["op_days"]
        efficiency = config["performance"]["efficiency"]
        turnaround_time_h = config["performance"]["turnaround_time_h"]
    else:          
        op_days    = myLumiModel.op_days
        efficiency = myLumiModel.efficiency
        turnaround_time_h  = myLumiModel.turnaround_time_h

    results["rel_time"] = (results["time"] - results["time"].iloc[0]).dt.total_seconds()
    lumi_int_pb = cumtrapz(results['lumi_avg'].values, x=results['rel_time'].values)*1e-40
    lumi_int_pb = np.insert(lumi_int_pb, 0, 0)
    results['lumi_int_pb_fill'] = lumi_int_pb

    if myLumiModel.include_lhcb:
        lumi_int_pb_ip8 = cumtrapz(results['lumi_LHCB'].apply(np.mean).values, x=results['rel_time'].values)*1e-40
        lumi_int_pb_ip8 = np.insert(lumi_int_pb_ip8, 0, 0)
        results['lumi_int_pb_fill_ip8'] = lumi_int_pb_ip8

    lumi_int_m_fill_taround = lumi_int_pb*1e40/(np.array(results['rel_time'].values) + turnaround_time_h*3600.0)#*24*3600.0
    lumi_int_m_day_taround = lumi_int_pb*1e40/(np.array(results['rel_time']) + turnaround_time_h*3600.0)*24*3600.0
    results["lumi_int_m_fill_taround"] = lumi_int_m_fill_taround
    results["lumi_int_m_day_taround"] = lumi_int_m_day_taround

    if myLumiModel.include_lhcb:
        lumi_int_m_fill_taround_ip8 = lumi_int_pb_ip8*1e40/(np.array(results['rel_time'].values) + turnaround_time_h*3600.0)#*24*3600.0
        lumi_int_m_day_taround_ip8 = lumi_int_pb_ip8*1e40/(np.array(results['rel_time']) + turnaround_time_h*3600.0)*24*3600.0
        results["lumi_int_m_fill_taround_ip8"] = lumi_int_m_fill_taround_ip8
        results["lumi_int_m_day_taround_ip8"] = lumi_int_m_day_taround_ip8


    idx = np.argmax(results["lumi_int_pb_fill"].values*1e40/(results["rel_time"].values + turnaround_time_h*3600))
    opt_fill_length = results["rel_time"].iloc[idx]/3600.0
    print('Optimal fill length: %sh'%(opt_fill_length))
    results['topt'] = opt_fill_length#/3600.0

    myint = results["lumi_int_pb_fill"]*1e40/(results["rel_time"].values + turnaround_time_h*3600)
    opt_int_lumi = results['lumi_int_pb_fill'][idx]

    if myLumiModel.include_lhcb:
        opt_int_lumi_ip8 = results['lumi_int_pb_fill_ip8'][idx]

    print('Opt int lumi [pb^-1]: %s'%opt_int_lumi)
    print('Yearly performance: %s fb-1/day, %s fb-1/year'%((lumi_int_m_day_taround[idx]*1e-43), (lumi_int_m_day_taround[idx]*1e-43)*op_days*efficiency))

    lumi_avg_xing = 1.0/(opt_int_lumi*1e40) * cumtrapz(np.array(results['lumi_avg'][0:idx])*np.array(results['xing'][0:idx])*2.0*1e-6, x=results['time'][0:idx])

    results['lumi_year'] = (lumi_int_m_day_taround[idx]*1e-43)*op_days*efficiency
    results['lumi_day']  = (lumi_int_m_day_taround[idx]*1e-43)
    if myLumiModel.include_lhcb:
        results['lumi_year_ip8'] = (lumi_int_m_day_taround_ip8[idx]*1e-43)*op_days*efficiency
        results['lumi_day_ip8']  = (lumi_int_m_day_taround_ip8[idx]*1e-43)
    results['Llevel0'] = myLumiModel.level_lumi_ip1

    try:
        level_time_ATLAS = results[np.isclose(results['lumi_ATLAS'].apply(lambda x:np.mean(x)), myLumiModel.level_lumi_ip1, rtol=1e-2)].iloc[-1]['rel_time']/3600.0
        level_time_CMS = results[np.isclose(results['lumi_CMS'].apply(lambda x:np.mean(x)), myLumiModel.level_lumi_ip5, rtol=1e-2)].iloc[-1]['rel_time']/3600.0
    except:
        level_time_ATLAS=0
        level_time_CMS = 0
    fig,ax = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(14,16))
    plt.sca(ax[0,0])
    plt.grid()
    int_hline        = results[np.isclose(results['rel_time'],opt_fill_length*3600.0)]['lumi_int_pb_fill'].values[0]#results[results['time'] == opt_fill_length*3600.0]['lumi_int_pb_fill'].values[0]

    plt.plot(results['rel_time']/3600.0, results['xing_ip1_y']*1e6/2., c='b',  marker='o', ms=1, lw=4)
    plt.plot(results['rel_time']/3600.0, results['xing_ip5_x']*1e6/2., c='r',  marker='o', ms=1, lw=4)
    plt.plot(results['rel_time']/3600.0, results['xing_ip1_x']*1e6/2., c='b',  marker='o', ms=1, lw=4)
    plt.plot(results['rel_time']/3600.0, results['xing_ip5_y']*1e6/2., c='r',  marker='o', ms=1, lw=4)
    plt.ylabel(r'$\rm \phi/2 \ (\mu rad)$')

    plt.sca(ax[0,1])
    plt.grid()

    plt.plot(results['rel_time']/3600.0, results['inten_B1']/1e11, c='b',  marker='o', ms=0)
    plt.ylabel(r'Intensity $\rm (\times 10^{11} \ ppb)$')
    plt.axhline(2.2,linestyle='--', c='dimgray', lw=2)


    plt.sca(ax[1,0])
    plt.grid()

    plt.plot(results['rel_time']/3600.0, results['emitx_B1'].apply(lambda x:np.mean(x))*1e6, c='c',  marker='o', ms=0, label=r'$\epsilon_{\rm n,x}$ B1')
    plt.plot(results['rel_time']/3600.0, results['emity_B1'].apply(lambda x:np.mean(x))*1e6, c='m',  marker='o', ms=0, label=r'$\epsilon_{\rm n,y}$ B1')
    plt.plot(results['rel_time']/3600.0, results['emitx_B2'].apply(lambda x:np.mean(x))*1e6, c='c',  marker='o', ms=0, linestyle='--', lw=2.5, label=r'$\epsilon_{\rm n,x}$ B2')
    plt.plot(results['rel_time']/3600.0, results['emity_B2'].apply(lambda x:np.mean(x))*1e6, c='m',  marker='o', ms=0, linestyle='--', lw=2.5, label=r'$\epsilon_{\rm n,y}$ B2')
    plt.ylabel(r'$\rm \epsilon_n \ (\mu m)$')
    plt.legend(fontsize=13, ncol=2)
    _emax = max(results[k].apply(lambda x:np.mean(x)).max() for k in ['emitx_B1','emity_B1','emitx_B2','emity_B2'])*1e6
    plt.ylim(1.3, max(3.5, _emax + 0.1))


    plt.sca(ax[1,1])
    plt.ylabel(r'Bunch length (cm)')
    plt.grid()
    plt.plot(results['rel_time']/3600.0, results['bl_s_B1'].apply(lambda x:np.mean(x))*clight/4.*1e2, c='g',  marker='o', ms=0)

    plt.sca(ax[2,0])
    plt.ylabel(r'$\rm \beta^*$ (m)')
    plt.grid()
    plt.plot(results['rel_time']/3600.0, results['betax_ip1'], c='orange',  marker='o', ms=0, label="IP1 H", lw=4)
    plt.plot(results['rel_time']/3600.0, results['betay_ip1'], c='darkorange',  marker='o', ms=0, label="IP1 V", lw=4)
    plt.legend()
    plt.axhline(1.0, linestyle='--', c='dimgray', lw=2)
    plt.axhline(0.15,linestyle='--', c='dimgray', lw=2)
    #plt.xlabel('Time (h)')

    plt.sca(ax[2,1])
    plt.ylabel('$\mathcal{L}$ ' + r'$\rm (10^{34} \ Hz/cm^{2})$')
    plt.grid()

    plt.plot(results['rel_time']/3600.0, np.array(results['lumi_ATLAS'].apply(lambda x:np.mean(x)))/1e38, c='b',  marker='o', ms=0, linestyle='-', lw=2)
    plt.plot(results['rel_time']/3600.0, np.array(results['lumi_CMS'].apply(lambda x:np.mean(x)))/1e38, c='r',  marker='o', ms=0, linestyle='-', lw=2)

    plt.axvline(level_time_ATLAS, c='b', linestyle='--', lw=2, label=r'$\rm t_{level,ATLAS} = %s \ h$' %round(level_time_ATLAS, 3))
    plt.axvline(level_time_CMS, c='r', linestyle='--', lw=2, label=r'$\rm t_{level,CMS} = %s \ h$' %round(level_time_CMS, 3))
    plt.legend(loc=3)


    ax2=ax[2,1].twinx()
    plt.plot(results['rel_time']/3600.0, np.array(results['pu_ATLAS']), c='b',  marker='o', ms=0, linestyle='-.', lw=1)
    plt.plot(results['rel_time']/3600.0, np.array(results['pu_CMS']), c='r',  marker='o', ms=0, linestyle='--', lw=1)
    plt.ylabel('Pileup (events)')
    ax2.yaxis.label.set_color('k')
    ax2.tick_params(axis='y', colors='k')

    # Initial y-limits for the axes
    initial_ylim_ax = ax[2, 1].get_ylim()
    initial_ylim_ax2 = ax2.get_ylim()

    # Scale factor based on the initial y-limits
    scale_factor = (initial_ylim_ax2[1] - initial_ylim_ax2[0]) / (initial_ylim_ax[1] - initial_ylim_ax[0])

    # Function to update the y-limits of one axis based on changes in the other
    def on_ylims_change(axx):
        if axx == ax[2, 1]:
            new_ylim = axx.get_ylim()
            expected_ylim_ax2 = (new_ylim[0] * scale_factor, new_ylim[1] * scale_factor)
            if ax2.get_ylim() != expected_ylim_ax2:
                ax2.set_ylim(expected_ylim_ax2)
        else:
            new_ylim = axx.get_ylim()
            expected_ylim_ax = (new_ylim[0] / scale_factor, new_ylim[1] / scale_factor)
            if ax[2, 1].get_ylim() != expected_ylim_ax:
                ax[2, 1].set_ylim(expected_ylim_ax)

    # Connect the event with the handler
    ax[2, 1].callbacks.connect('ylim_changed', on_ylims_change)
    ax2.callbacks.connect('ylim_changed', on_ylims_change)


    #plt.axhline(myLumiModel.max_pileup_ip5, linestyle='--', c='r', lw=1)

    plt.sca(ax[3,0])
    plt.grid()

    plt.xlabel('Time in SB (h)')
    plt.ylabel(r'$\int_{0}^{t}  \mathcal{L}$d$\tau$'+r' $\rm (pb^{-1})$')
    plt.plot(results['rel_time']/3600.0, np.array(results['lumi_int_pb_fill']), c='tab:blue',  marker='o', ms=0)
    plt.axhline(int_hline, c='k', linestyle='--', lw=2) 
    label1 = r'$\rm \mathcal{L}_{int}(t_{opt}) = %s \ fb^{-1}$' %(round(int_hline/1e3, 3))
    label1 +="\n"
    label1 += r'$\rm t_{opt} = %s \ h$' %(round(opt_fill_length, 3))
    #label1=''

    plt.axvline(opt_fill_length, c='k', linestyle='--', lw=2, label=label1)
    plt.legend(fontsize=20)

    plt.sca(ax[3,1])
    plt.grid()

    plt.ylabel(r'$\frac{\int_{0}^{t}  \mathcal{L}\rm{d}\tau}{t+\tau_0}$'+r'$\rm (fb^{-1}/day)$')
    plt.plot(results['rel_time']/3600.0, np.array(results['lumi_int_m_day_taround'])*1e-43, c='tab:purple',  marker='o', ms=0)

    label1 = r'$\rm t_{opt} = %s \ h$' %(round(opt_fill_length, 3)) +  r", $\tau_0=$" + f"{turnaround_time_h} h"
    label1 +="\n"
    label1 +=  str(round(lumi_int_m_day_taround[idx]*1e-43, 2)) + r" $\rm fb^{-1}/day$"
    label1 +="\n"
    label1 += str(round((lumi_int_m_day_taround[idx]*1e-43)*op_days*efficiency, 2)) + r" $\rm fb^{-1}/year $"+ f" for {op_days} days" + r", $\eta$=" + f"{efficiency}"

    plt.axvline(opt_fill_length, c='k', linestyle='--', lw=2, label=label1)
    plt.legend()
    plt.xlabel('Time in SB (h)')

    return results, fig, ax, opt_fill_length, level_time_ATLAS, level_time_CMS


def plot_ip15_separation(results, myLumiModel, emit_0=2.5, flag_virtual=False):
    fig, ax = plt.subplots(ncols=2, figsize=(14,8))

    plt.sca(ax[0])
    plt.grid()
    plt.plot(results['rel_time']/3600.0, results['lumi_ATLAS']/1e38, c='b', lw=2, label="ATLAS")
    plt.plot(results['rel_time']/3600.0, results['lumi_CMS']/1e38, c='r', lw=2, label="CMS")



    plt.legend()

    if flag_virtual:
        max_virtual = myLumiModel.level_virtual_lumi
        tolerance = myLumiModel.virtual_lumi_tol_percentage/100.0
        lower_bound = max_virtual - (max_virtual * tolerance)
        upper_bound = max_virtual 
        plt.axhspan(lower_bound/1e38, upper_bound/1e38, alpha=0.2, color='gray')

        plt.plot(results['rel_time']/3600.0, results['virtual_lumi_ATLAS_2um']/1e38, c='lime', lw=1, label="Virtual ATLAS, 2um")
        plt.plot(results['rel_time']/3600.0, results['virtual_lumi_CMS_2um']/1e38, c='orange', lw=2, label="Virtual CMS, 2um")

        plt.plot(results['rel_time']/3600.0, results['virtual_lumi_ATLAS']/1e38, c='navy', lw=1, label="Virtual ATLAS")
        plt.plot(results['rel_time']/3600.0, results['virtual_lumi_CMS']/1e38, c='firebrick', lw=2, label="Virtual CMS")

    max_virtual = myLumiModel.level_lumi_ip1
    tolerance = myLumiModel.lumi_tol_percentage/100.0
    lower_bound = max_virtual - (max_virtual * tolerance)
    upper_bound = max_virtual 
    plt.axhspan(lower_bound/1e38, upper_bound/1e38, alpha=0.2, color='b')


    max_virtual = myLumiModel.level_lumi_ip5
    tolerance = myLumiModel.lumi_tol_percentage/100.0
    lower_bound = max_virtual - (max_virtual * tolerance)
    upper_bound = max_virtual 
    plt.axhspan(lower_bound/1e38, upper_bound/1e38, alpha=0.2, color='r')

    plt.xlabel('Time (h)')
    plt.ylabel(r'$\mathcal{L}$ ' + r'$\rm (10^{34} \ Hz/cm^{2})$')

    plt.sca(ax[1])
    plt.grid()
    sigma_x = np.sqrt(emit_0*1e-6/myLumiModel.gamma*results["betax_ip1"])
    sigma_y = np.sqrt(emit_0*1e-6/myLumiModel.gamma*results["betax_ip1"])
    plt.plot(results['rel_time']/3600.0, 2.0*results['on_sep1_x']/sigma_x, c='b', lw=2, label="ATLAS, H")
    plt.plot(results['rel_time']/3600.0, 2.0*results['on_sep1_y']/sigma_y, c='b', lw=2, label="ATLAS, V", linestyle='--')
    sigma_x = np.sqrt(emit_0*1e-6/myLumiModel.gamma*results["betax_ip5"])
    sigma_y = np.sqrt(emit_0*1e-6/myLumiModel.gamma*results["betax_ip5"])
    plt.plot(results['rel_time']/3600.0, 2.0*results['on_sep5_x']/sigma_x, c='r', lw=2, label="CMS, H")
    plt.plot(results['rel_time']/3600.0, 2.0*results['on_sep5_y']/sigma_x, c='r', lw=2, label="CMS, V", linestyle='--')
    plt.xlabel('Time(h)')
    plt.legend()

    plt.ylabel(r'Full separation $(\sigma)$')
    return results, fig, ax


def cmp_longitudinal_luminous_region(results, myLumiModel, verbose=False, nb=None,
                                      ccrf=400e6, R12=46.66133302853659, CC_voltage=-3.4e6):

    #sigma_z_list = np.linspace(-0.09, 0.09, 100)
    sigma_z_list = np.linspace(-0.2, 0.2, 200)
    current_row = results.iloc[0]
    res = {}
    res["ip1"] = np.zeros((len(results),1,5))
    res["ip5"] = np.zeros((len(results),1,5))
    res_density = {}
    res_pu_density = {}
    res_density["ip1"] = []
    res_density["ip5"] = []
    res_pu_density["ip1"] = []
    res_pu_density["ip5"] = []

    ips = [1,5]
    experiments = ["ATLAS", "CMS"]
    if myLumiModel.include_lhcb:
        res["ip8"] = np.zeros((len(results),1,5))
        res_density["ip8"] = []
        res_pu_density["ip8"] = []
        ips = [1,5,8]
        experiments = ["ATLAS", "CMS", "LHCb"]
    for counter, (ip, experiment) in enumerate(zip(ips, experiments)):
        for counter2, (index, current_row) in enumerate(results.iterrows()):
            for i in range(len(current_row["emitx_B1"])):
                if myLumiModel.include_cc and ip != 8:
                    if getattr(myLumiModel, f"{experiment}_crossing") == "H":
                            myR12, myV_x = R12, CC_voltage
                            myR34, myV_y = 0, 0

                    else:
                            myR34, myV_y = R12, CC_voltage
                            myR12, myV_x = 0, 0

                    if i== 0 and counter2 == 0:
                        print("CC included!")
                        print("experiment: ", experiment)
                        print("R12: ", myR12)
                        print("V_x: ", myV_x)
                        print("R34: ", myR34)
                        print("V_y: ", myV_y)
                                        
                else:
                    if i == 0 and counter2 == 0:
                        print("CC not included!")
                    myV_x, myV_y, myR12, myR34 = 0, 0, 0, 0

                if nb is None:
                    if ip==1 or ip==5:
                        nb = myLumiModel.number_of_bunches_coll_ip15
                    elif ip==8:
                        nb = myLumiModel.number_of_bunches_coll_ip8
                test = lumi_integrals.longitudinal_luminous_region(sigma_z_list,
                    f=11245.5, nb=nb, 
                    N1=current_row["inten_B1"][i], N2=current_row["inten_B2"][i], 
                    x_1=current_row[f"on_sep{ip}_x"], x_2=-current_row[f"on_sep{ip}_x"], 
                    y_1=current_row[f"on_sep{ip}_y"], y_2=-current_row[f"on_sep{ip}_y"], 
                    px_1=current_row[f"xing_ip{ip}_x"]/2., px_2=-current_row[f"xing_ip{ip}_x"]/2., 
                    py_1=current_row[f"xing_ip{ip}_y"]/2., py_2=-current_row[f"xing_ip{ip}_y"]/2., 
                    energy_tot1=myLumiModel.En*1e9, energy_tot2=myLumiModel.En*1e9, 
                    deltap_p0_1=0, deltap_p0_2=0, 
                    epsilon_x1=current_row["emitx_B1"][i], epsilon_x2=current_row["emitx_B2"][i], 
                    epsilon_y1=current_row["emity_B1"][i], epsilon_y2=current_row["emity_B2"][i], 
                    sigma_z1=current_row["bl_s_B1"][i]*clight/4., sigma_z2=current_row["bl_s_B2"][i]*clight/4., 
                    beta_x1=current_row[f"betax_ip{ip}"], beta_x2=current_row[f"betax_ip{ip}"], 
                    beta_y1=current_row[f"betay_ip{ip}"], beta_y2=current_row[f"betay_ip{ip}"], 
                    alpha_x1=0, alpha_x2=0, 
                    alpha_y1=0, alpha_y2=0, 
                    dx_1=0, dx_2=0, dy_1=0, dy_2=0, 
                    dpx_1=0, dpx_2=0, dpy_1=0, dpy_2=0, 
                    CC_V_x_1=getattr(myLumiModel,'cc_b1_fraction',1.0)*myV_x, CC_f_x_1=ccrf, CC_phase_x_1=0, 
                    CC_V_x_2=-getattr(myLumiModel,'cc_b2_fraction',1.0)*myV_x, CC_f_x_2=ccrf, CC_phase_x_2=0, 
                    CC_V_y_1=getattr(myLumiModel,'cc_b1_fraction',1.0)*myV_y, CC_f_y_1=ccrf, CC_phase_y_1=0, 
                    CC_V_y_2=-getattr(myLumiModel,'cc_b2_fraction',1.0)*myV_y, CC_f_y_2=ccrf, CC_phase_y_2=0, 
                    R12_1=myR12, R22_1=0, R34_1=myR34, R44_1=0, 
                    R12_2=myR12, R22_2=0, R34_2=myR34, R44_2=0, 
                    verbose=verbose, sigma_integration=3.0)

                # longitudinal density
                #res[f"ip{ip}"][index, 0, 0] = test[0]


                res_density[f"ip{ip}"].append(test[0])
                # rms
                res[f"ip{ip}"][index, 0, 0] = test[1]
                # std_dev
                res[f"ip{ip}"][index, 0, 1] = test[2]
                # total luminosity
                res[f"ip{ip}"][index, 0, 2] = test[3]
                # peak pileup density
                res[f"ip{ip}"][index, 0, 3] = test[4]
                # pileup density
                #res[f"ip{ip}"][index, 0, 5] = test[5]
                res_pu_density[f"ip{ip}"].append(test[5])
                # mean
                res[f"ip{ip}"][index, 0, 4] = test[6]



    results["long_luminous_ip5_rms"] = np.mean(res["ip5"][:,:,0]*1e3, axis=1)
    results["long_luminous_ip1_rms"] = np.mean(res["ip1"][:,:,0]*1e3, axis=1)
    results["long_luminous_avg_rms"] = np.mean([np.mean(res["ip5"][:,:,0]*1e3, axis=1), np.mean(res["ip1"][:,:,0]*1e3, axis=1)], axis=0)
    
    results["long_luminous_ip5_std"] = np.mean(res["ip5"][:,:,1]*1e3, axis=1)
    results["long_luminous_ip1_std"] = np.mean(res["ip1"][:,:,1]*1e3, axis=1)
    results["long_luminous_avg_std"] = np.mean([np.mean(res["ip5"][:,:,1]*1e3, axis=1), np.mean(res["ip1"][:,:,1]*1e3, axis=1)], axis=0)
    
    results["mean_z_ip5"] = np.mean(res["ip5"][:,:,4], axis=1)
    results["mean_z_ip1"] = np.mean(res["ip1"][:,:,4], axis=1)
    results["mean_z_avg"] = np.mean([np.mean(res["ip5"][:,:,4], axis=1), np.mean(res["ip1"][:,:,4], axis=1)], axis=0)

    results["peak_pileup_density_ip5"] = np.mean(res["ip5"][:,:,3]*1e-3, axis=1)
    results["peak_pileup_density_ip1"] = np.mean(res["ip1"][:,:,3]*1e-3, axis=1)
    results["peak_pileup_density_avg"] = np.mean([np.mean(res["ip5"][:,:,3]*1e-3, axis=1), np.mean(res["ip1"][:,:,3]*1e-3, axis=1)], axis=0)

    results["lumi_from_density_ip1"] = np.mean(res["ip1"][:,:,2], axis=1)
    results["lumi_from_density_ip5"] = np.mean(res["ip5"][:,:,2], axis=1)
    results["lumi_from_density_avg"] = np.mean([np.mean(res["ip5"][:,:,2], axis=1), np.mean(res["ip1"][:,:,2], axis=1)], axis=0)

    if myLumiModel.include_lhcb:
        results["long_luminous_ip8_rms"] = np.mean(res["ip8"][:,:,0]*1e3, axis=1)
        results["long_luminous_ip8_std"] = np.mean(res["ip8"][:,:,1]*1e3, axis=1)
        results["peak_pileup_density_ip8"] = np.mean(res["ip8"][:,:,3]*1e-3, axis=1)
        results["lumi_from_density_ip8"] = np.mean(res["ip8"][:,:,2], axis=1)
        results["mean_z_ip8"] = np.mean(res["ip8"][:,:,4], axis=1)
    return (results, res_density, res_pu_density)


def plot_longitudinal_luminous_region(results, myLumiModel, emit_0=2.0):
    fig, ax = plt.subplots(ncols=3, figsize=(18,8))

    plt.sca(ax[0])
    plt.grid()
    plt.plot(results['rel_time']/3600.0, results['lumi_ATLAS']/1e38, c='b', lw=2, label="ATLAS")
    plt.plot(results['rel_time']/3600.0, results['lumi_CMS']/1e38, c='r', lw=2, label="CMS")


    plt.legend()

    plt.xlabel('Time (h)')
    plt.ylabel(r'$\mathcal{L}$ ' + r'$\rm (10^{34} \ Hz/cm^{2})$')

    plt.sca(ax[1])
    plt.grid()
    plt.plot(results["rel_time"]/3600.0, results["long_luminous_avg_std"], c="g", lw=4)
    plt.plot(results["rel_time"]/3600.0, results["long_luminous_ip1_std"], c="b", lw=2, linestyle='--')
    plt.plot(results["rel_time"]/3600.0, results["long_luminous_ip5_std"], c="r", lw=2, linestyle='--')
    plt.xlabel('Time(h)')
    plt.ylabel("Longitudinal luminous region (mm)")

    plt.sca(ax[2])
    plt.grid()
    sigma_x = np.sqrt(emit_0*1e-6/myLumiModel.gamma*results["betax_ip1"])
    sigma_y = np.sqrt(emit_0*1e-6/myLumiModel.gamma*results["betay_ip1"])
    plt.plot(results['rel_time']/3600.0, 2.0*results['on_sep1_x']/sigma_x, c='b', lw=3, label="ATLAS, H")
    plt.plot(results['rel_time']/3600.0, 2.0*results['on_sep1_y']/sigma_y, c='b', lw=3, label="ATLAS, V", linestyle='--')
    #plt.plot(results['rel_time']/3600.0, 2.0*results['on_sep5']/sigma, c='r', lw=2, label="CMS")
    plt.xlabel('Time (h)')
    plt.ylabel(r'Vertical full separation in ATLAS $(\sigma)$')
    return fig, ax


def plot_ip8(results, myLumiModel, level_time_ip15=0, opt_fill_length=0, emit_0=2.0, config="config.yaml"):
    if myLumiModel is None:
        import yaml
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
        op_days = config["performance"]["op_days"]
        efficiency = config["performance"]["efficiency"]
        turnaround_time_h = config["performance"]["turnaround_time_h"]
    else:          
        op_days    = myLumiModel.op_days
        efficiency = myLumiModel.efficiency
        turnaround_time_h  = myLumiModel.turnaround_time_h

    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(14,16), sharex=True)


    plt.sca(ax[0,0])
    plt.grid()
    plt.plot(results["rel_time"].values/3600.0, results["pu_LHCB"])
    plt.ylabel("LHCb pileup")
    ax2 = ax[0,0].twinx()
    plt.plot(results["rel_time"].values/3600.0, results["lumi_LHCB"].apply(lambda x:np.mean(x))/1e38, c='k', lw=0.1)
    plt.ylabel(r'$\mathcal{L}$ ' + r'$\rm (10^{34} \ Hz/cm^{2})$')
    plt.sca(ax[0,0])

    current_pu = np.max(results['pu_LHCB'])
    level_time_ip8 = results[np.isclose(results['pu_LHCB'].apply(lambda x:np.mean(x)), current_pu, rtol=1e-3)].iloc[-1]['rel_time']/3600.0
    mytext = r'$\rm t_{level}^{IP8} = %s \ h$' %round(level_time_ip8, 3)
    plt.axvline(level_time_ip8, c="k", linestyle='--', lw=1, label=mytext)

    mytext = r'$\rm t_{level}^{IP1/5} = %s \ h$' %round(level_time_ip15, 3)
    plt.axvline(level_time_ip15, c="magenta", linestyle='--', lw=1, label=mytext)

    mytext = r'$\rm t_{opt}^{IP1/5} = %s \ h$' %round(opt_fill_length, 3)
    plt.axvline(opt_fill_length, c="cyan", linestyle='--', lw=1, label=mytext)

    
    plt.legend()
    # Initial y-limits for the axes
    initial_ylim_ax = ax[0, 0].get_ylim()
    initial_ylim_ax2 = ax2.get_ylim()

    # Scale factor based on the initial y-limits
    scale_factor = (initial_ylim_ax2[1] - initial_ylim_ax2[0]) / (initial_ylim_ax[1] - initial_ylim_ax[0])

    # Function to update the y-limits of one axis based on changes in the other
    def on_ylims_change(axx):
        if axx == ax[0, 0]:
            new_ylim = axx.get_ylim()
            expected_ylim_ax2 = (new_ylim[0] * scale_factor, new_ylim[1] * scale_factor)
            if ax2.get_ylim() != expected_ylim_ax2:
                ax2.set_ylim(expected_ylim_ax2)
        else:
            new_ylim = axx.get_ylim()
            expected_ylim_ax = (new_ylim[0] / scale_factor, new_ylim[1] / scale_factor)
            if ax[0, 0].get_ylim() != expected_ylim_ax:
                ax[0, 0].set_ylim(expected_ylim_ax)

    # Connect the event with the handler
    ax[0, 0].callbacks.connect('ylim_changed', on_ylims_change)
    ax2.callbacks.connect('ylim_changed', on_ylims_change)

    plt.ylim(current_pu-2, current_pu+2)

    plt.sca(ax[0,1])
    plt.plot(results["rel_time"].values/3600.0, results["betax_ip8"])
    plt.plot(results["rel_time"].values/3600.0, results["betay_ip8"])
    plt.grid()
    plt.ylabel(r"$\beta_{\rm IP8} \ (m)$")



    plt.sca(ax[1,0])
    plt.grid()
    sigma_ip8 = np.sqrt(emit_0*1e-6/myLumiModel.gamma*results["betax_ip8"])
    plt.plot(results["rel_time"].values/3600.0, 2.0*results["on_sep8_r"]/sigma_ip8, label="Radial")
    plt.plot(results["rel_time"].values/3600.0, 2.0*results["on_sep8_x"]/sigma_ip8, label="Horizontal")
    plt.plot(results["rel_time"].values/3600.0, 2.0*results["on_sep8_y"]/sigma_ip8, label="Vertical")
    plt.xlabel("Time (h)")
    plt.legend()
    plt.ylabel(r"Full separation $(\sigma)$")


    plt.sca(ax[1,1])
    plt.grid()
    plt.plot(results["rel_time"].values/3600.0, results["xing_ip8_r"]*1e6/2, label="Radial")
    plt.plot(results["rel_time"].values/3600.0, results["xing_ip8_x"]*1e6/2, label="Horizontal")
    plt.plot(results["rel_time"].values/3600.0, results["xing_ip8_y"]*1e6/2, label="Vertical")
    plt.xlabel("Time (h)")
    plt.legend()
    plt.ylabel(r"$\phi/2 (\mu rad)$")

    plt.sca(ax[2,0])
    plt.grid()

    plt.xlabel('Time in SB (h)')
    plt.ylabel(r'$\int_{0}^{t}  \mathcal{L}$d$\tau$'+r' $\rm (pb^{-1})$')
    plt.plot(results['rel_time']/3600.0, np.array(results['lumi_int_pb_fill_ip8']), c='tab:blue',  marker='o', ms=0)
    #plt.axhline(int_hline, c='k', linestyle='--', lw=2) 
    #label1 = r'$\rm \mathcal{L}_{int}(t_{opt}) = %s \ fb^{-1}$' %(round(int_hline/1e3, 3))
    #label1 +="\n"
    #label1 += r'$\rm t_{opt} = %s \ h$' %(round(opt_fill_length, 3))
    #label1=''
    plt.axvline(opt_fill_length, c="cyan", linestyle='--', lw=1)

    #plt.axvline(opt_fill_length, c='k', linestyle='--', lw=2, label=label1)
    plt.legend(fontsize=20)

    plt.sca(ax[2,1])
    plt.grid()

    plt.ylabel(r'$\frac{\int_{0}^{t}  \mathcal{L}\rm{d}\tau}{t+\tau_0}$'+r'$\rm (fb^{-1}/day)$')
    plt.plot(results['rel_time']/3600.0, np.array(results['lumi_int_m_day_taround_ip8'])*1e-43, c='tab:purple',  marker='o', ms=0)

    #label1 = r'$\rm t_{opt} = %s \ h$' %(round(opt_fill_length, 3)) + r", $\tau_0=\rm 3.5 \ h$"
    #label1 +="\n"
    #label1 +=  str(round(lumi_int_m_day_taround[idx]*1e-43, 2)) + r" $\rm fb^{-1}/day$"
    #label1 +="\n"
    #label1 += str(round((lumi_int_m_day_taround[idx]*1e-43)*op_days*efficiency, 2)) + r"$\rm fb^{-1}/year $"+ f" for {op_days} days" + r", $\eta$=0.5"
    
    label1 = r'$\rm t_{opt} = %s \ h$' %(round(opt_fill_length, 3)) + r", $\tau_0=$" + f"{turnaround_time_h} h"
    label1 +="\n"
    label1 +=  str(round(results['lumi_day_ip8'].iloc[0], 2)) + r" $\rm fb^{-1}/day$"
    label1 +="\n"
    label1 += str(round(results['lumi_year_ip8'].iloc[0], 2)) + r" $\rm fb^{-1}/year $"+ f" for {op_days} days" + r", $\eta$=" + f"{efficiency}"

    plt.axvline(opt_fill_length, c='cyan', linestyle='--', lw=2, label=label1)

    
    #plt.axvline(opt_fill_length, c="cyan", linestyle='--', lw=1)
    #plt.axvline(opt_fill_length, c='k', linestyle='--', lw=2, label=label1)
    plt.legend()
    plt.xlabel('Time in SB (h)')

    return fig, ax

def plot_ip8_longitudinal_luminous_region(results, opt_fill_length=0):
    fig, ax = plt.subplots(ncols=1, figsize=(9,7))

    plt.grid()
    plt.plot(results["rel_time"]/3600.0, results["long_luminous_ip8_std"], c="k", lw=4)
    plt.xlabel('Time (h)')
    plt.ylabel("Longitudinal luminous region (mm)", fontsize=22)
    plt.axvline(opt_fill_length, c="c", linestyle='--', lw=1)
    return fig, ax


def cmp_transverse_luminous_region(results, myLumiModel, direction="both", verbose=False, nb=None,
                                    ccrf=400e6, R12=46.66133302853659, CC_voltage=-3.4e6,
                                    n_sigma=6, n_points=60):
    """Transverse analogue of cmp_longitudinal_luminous_region.

    Computes the horizontal and/or vertical luminous region size (rms/std of
    the luminosity density as a function of the transverse offset from the
    nominal collision axis) at IP1 (ATLAS), IP5 (CMS) and, if enabled, IP8
    (LHCb), for every time step stored in `results`.

    direction : "x", "y" or "both" (default) selects which transverse
        plane(s) to compute.
    n_sigma, n_points : the position scan at each time step/IP is built as
        n_points values spanning +/- n_sigma times the *local* beam size
        (unlike the bunch length, the transverse beam size changes a lot
        with beta* over a fill, so a fixed absolute grid like the one used
        for the longitudinal z-scan would not adequately resolve it at
        every time step).

    Adds to `results` (for each computed direction "x"/"y", here written as
    "u"): trans_luminous_u_ip{1,5,8}_rms/std (in um), mean_u_ip{1,5,8} (in
    m), peak_pileup_density_u_ip{1,5,8}, lumi_from_density_u_ip{1,5,8}, plus
    the corresponding *_avg combining IP1 and IP5.

    Returns (results, res_density, res_pu_density) where res_density and
    res_pu_density are dictionaries keyed by direction, each containing the
    same structure returned by cmp_longitudinal_luminous_region's
    res_density/res_pu_density.
    """

    directions = ["x", "y"] if direction == "both" else [direction]

    ips = [1, 5]
    experiments = ["ATLAS", "CMS"]
    if myLumiModel.include_lhcb:
        ips = [1, 5, 8]
        experiments = ["ATLAS", "CMS", "LHCb"]

    res_density_all = {}
    res_pu_density_all = {}

    for mydir in directions:
        res = {f"ip{ip}": np.zeros((len(results), 1, 5)) for ip in ips}
        res_density = {f"ip{ip}": [] for ip in ips}
        res_pu_density = {f"ip{ip}": [] for ip in ips}

        for counter, (ip, experiment) in enumerate(zip(ips, experiments)):
            for counter2, (index, current_row) in enumerate(results.iterrows()):
                for i in range(len(current_row["emitx_B1"])):
                    if myLumiModel.include_cc and ip != 8:
                        if getattr(myLumiModel, f"{experiment}_crossing") == "H":
                            myR12, myV_x = R12, CC_voltage
                            myR34, myV_y = 0, 0
                        else:
                            myR34, myV_y = R12, CC_voltage
                            myR12, myV_x = 0, 0

                        if i == 0 and counter2 == 0:
                            print(f"CC included! ({mydir})")
                            print("experiment: ", experiment)
                    else:
                        if i == 0 and counter2 == 0:
                            print(f"CC not included! ({mydir})")
                        myV_x, myV_y, myR12, myR34 = 0, 0, 0, 0

                    if nb is None:
                        if ip == 1 or ip == 5:
                            mynb = myLumiModel.number_of_bunches_coll_ip15
                        elif ip == 8:
                            mynb = myLumiModel.number_of_bunches_coll_ip8
                    else:
                        mynb = nb

                    # local beam size at this IP/time step/direction, used to set
                    # a sensible scan range (emitx_B1/emity_B1 store the
                    # *normalized* emittance in m, matching the epsilon_x1/
                    # epsilon_y1 convention used by lumi_integrals, whose sigma()
                    # helper divides by betagamma internally -- so we do the same
                    # here to get the physical beam size)
                    beta_u = current_row[f"beta{mydir}_ip{ip}"]
                    emit_u_norm = 0.5 * (current_row[f"emit{mydir}_B1"][i] + current_row[f"emit{mydir}_B2"][i])
                    sigma_u = np.sqrt(max(emit_u_norm * beta_u / myLumiModel.gamma, 1e-30))
                    pos_list = np.linspace(-n_sigma * sigma_u, n_sigma * sigma_u, n_points)

                    test = lumi_integrals.transverse_luminous_region(pos_list, direction=mydir,
                        f=11245.5, nb=mynb,
                        N1=current_row["inten_B1"][i], N2=current_row["inten_B2"][i],
                        x_1=current_row[f"on_sep{ip}_x"], x_2=-current_row[f"on_sep{ip}_x"],
                        y_1=current_row[f"on_sep{ip}_y"], y_2=-current_row[f"on_sep{ip}_y"],
                        px_1=current_row[f"xing_ip{ip}_x"]/2., px_2=-current_row[f"xing_ip{ip}_x"]/2.,
                        py_1=current_row[f"xing_ip{ip}_y"]/2., py_2=-current_row[f"xing_ip{ip}_y"]/2.,
                        energy_tot1=myLumiModel.En*1e9, energy_tot2=myLumiModel.En*1e9,
                        deltap_p0_1=0, deltap_p0_2=0,
                        epsilon_x1=current_row["emitx_B1"][i], epsilon_x2=current_row["emitx_B2"][i],
                        epsilon_y1=current_row["emity_B1"][i], epsilon_y2=current_row["emity_B2"][i],
                        sigma_z1=current_row["bl_s_B1"][i]*clight/4., sigma_z2=current_row["bl_s_B2"][i]*clight/4.,
                        beta_x1=current_row[f"betax_ip{ip}"], beta_x2=current_row[f"betax_ip{ip}"],
                        beta_y1=current_row[f"betay_ip{ip}"], beta_y2=current_row[f"betay_ip{ip}"],
                        alpha_x1=0, alpha_x2=0,
                        alpha_y1=0, alpha_y2=0,
                        dx_1=0, dx_2=0, dy_1=0, dy_2=0,
                        dpx_1=0, dpx_2=0, dpy_1=0, dpy_2=0,
                        CC_V_x_1=getattr(myLumiModel,'cc_b1_fraction',1.0)*myV_x, CC_f_x_1=ccrf, CC_phase_x_1=0,
                        CC_V_x_2=-getattr(myLumiModel,'cc_b2_fraction',1.0)*myV_x, CC_f_x_2=ccrf, CC_phase_x_2=0,
                        CC_V_y_1=getattr(myLumiModel,'cc_b1_fraction',1.0)*myV_y, CC_f_y_1=ccrf, CC_phase_y_1=0,
                        CC_V_y_2=-getattr(myLumiModel,'cc_b2_fraction',1.0)*myV_y, CC_f_y_2=ccrf, CC_phase_y_2=0,
                        R12_1=myR12, R22_1=0, R34_1=myR34, R44_1=0,
                        R12_2=myR12, R22_2=0, R34_2=myR34, R44_2=0,
                        verbose=verbose, sigma_integration=3.0)

                    res_density[f"ip{ip}"].append(test[0])
                    # rms
                    res[f"ip{ip}"][index, 0, 0] = test[1]
                    # std_dev
                    res[f"ip{ip}"][index, 0, 1] = test[2]
                    # total luminosity
                    res[f"ip{ip}"][index, 0, 2] = test[3]
                    # peak pileup density
                    res[f"ip{ip}"][index, 0, 3] = test[4]
                    res_pu_density[f"ip{ip}"].append(test[5])
                    # mean
                    res[f"ip{ip}"][index, 0, 4] = test[6]

        results[f"trans_luminous_{mydir}_ip5_rms"] = np.mean(res["ip5"][:,:,0]*1e6, axis=1)
        results[f"trans_luminous_{mydir}_ip1_rms"] = np.mean(res["ip1"][:,:,0]*1e6, axis=1)
        results[f"trans_luminous_{mydir}_avg_rms"] = np.mean([np.mean(res["ip5"][:,:,0]*1e6, axis=1), np.mean(res["ip1"][:,:,0]*1e6, axis=1)], axis=0)

        results[f"trans_luminous_{mydir}_ip5_std"] = np.mean(res["ip5"][:,:,1]*1e6, axis=1)
        results[f"trans_luminous_{mydir}_ip1_std"] = np.mean(res["ip1"][:,:,1]*1e6, axis=1)
        results[f"trans_luminous_{mydir}_avg_std"] = np.mean([np.mean(res["ip5"][:,:,1]*1e6, axis=1), np.mean(res["ip1"][:,:,1]*1e6, axis=1)], axis=0)

        results[f"mean_{mydir}_ip5"] = np.mean(res["ip5"][:,:,4], axis=1)
        results[f"mean_{mydir}_ip1"] = np.mean(res["ip1"][:,:,4], axis=1)
        results[f"mean_{mydir}_avg"] = np.mean([np.mean(res["ip5"][:,:,4], axis=1), np.mean(res["ip1"][:,:,4], axis=1)], axis=0)

        results[f"peak_pileup_density_{mydir}_ip5"] = np.mean(res["ip5"][:,:,3]*1e-6, axis=1)
        results[f"peak_pileup_density_{mydir}_ip1"] = np.mean(res["ip1"][:,:,3]*1e-6, axis=1)
        results[f"peak_pileup_density_{mydir}_avg"] = np.mean([np.mean(res["ip5"][:,:,3]*1e-6, axis=1), np.mean(res["ip1"][:,:,3]*1e-6, axis=1)], axis=0)

        results[f"lumi_from_density_{mydir}_ip1"] = np.mean(res["ip1"][:,:,2], axis=1)
        results[f"lumi_from_density_{mydir}_ip5"] = np.mean(res["ip5"][:,:,2], axis=1)
        results[f"lumi_from_density_{mydir}_avg"] = np.mean([np.mean(res["ip5"][:,:,2], axis=1), np.mean(res["ip1"][:,:,2], axis=1)], axis=0)

        if myLumiModel.include_lhcb:
            results[f"trans_luminous_{mydir}_ip8_rms"] = np.mean(res["ip8"][:,:,0]*1e6, axis=1)
            results[f"trans_luminous_{mydir}_ip8_std"] = np.mean(res["ip8"][:,:,1]*1e6, axis=1)
            results[f"peak_pileup_density_{mydir}_ip8"] = np.mean(res["ip8"][:,:,3]*1e-6, axis=1)
            results[f"lumi_from_density_{mydir}_ip8"] = np.mean(res["ip8"][:,:,2], axis=1)
            results[f"mean_{mydir}_ip8"] = np.mean(res["ip8"][:,:,4], axis=1)

        res_density_all[mydir] = res_density
        res_pu_density_all[mydir] = res_pu_density

    return results, res_density_all, res_pu_density_all


def plot_transverse_luminous_region(results, myLumiModel, direction="both"):
    """Transverse analogue of plot_longitudinal_luminous_region: plots the
    horizontal and/or vertical luminous region size (in um) vs. time,
    requires cmp_transverse_luminous_region to have been run first."""
    directions = ["x", "y"] if direction == "both" else [direction]
    fig, ax = plt.subplots(ncols=len(directions), figsize=(9*len(directions), 8), squeeze=False)
    ax = ax[0]

    for i, mydir in enumerate(directions):
        plt.sca(ax[i])
        plt.grid()
        plt.plot(results["rel_time"]/3600.0, results[f"trans_luminous_{mydir}_avg_std"], c="g", lw=4, label="Average")
        plt.plot(results["rel_time"]/3600.0, results[f"trans_luminous_{mydir}_ip1_std"], c="b", lw=2, linestyle='--', label="ATLAS")
        plt.plot(results["rel_time"]/3600.0, results[f"trans_luminous_{mydir}_ip5_std"], c="r", lw=2, linestyle='--', label="CMS")
        plt.xlabel('Time (h)')
        label = "Horizontal" if mydir == "x" else "Vertical"
        plt.ylabel(f"{label} luminous region " + r"($\rm \mu m$)")
        plt.legend()

    return fig, ax


def plot_ip8_transverse_luminous_region(results, opt_fill_length=0, direction="both"):
    """Transverse analogue of plot_ip8_longitudinal_luminous_region."""
    directions = ["x", "y"] if direction == "both" else [direction]
    fig, ax = plt.subplots(ncols=len(directions), figsize=(9*len(directions), 7), squeeze=False)
    ax = ax[0]

    for i, mydir in enumerate(directions):
        plt.sca(ax[i])
        plt.grid()
        plt.plot(results["rel_time"]/3600.0, results[f"trans_luminous_{mydir}_ip8_std"], c="k", lw=4)
        plt.xlabel('Time (h)')
        label = "Horizontal" if mydir == "x" else "Vertical"
        plt.ylabel(f"{label} luminous region " + r"($\rm \mu m$)", fontsize=22)
        plt.axvline(opt_fill_length, c="c", linestyle='--', lw=1)

    return fig, ax


def plot_ip8_pileup_density(results, opt_fill_length=0):
    fig, ax = plt.subplots(ncols=1, figsize=(9,7))

    plt.grid()
    plt.plot(results["rel_time"]/3600.0, results["peak_pileup_density_ip8"], c="k", lw=4)
    plt.xlabel('Time (h)')
    plt.ylabel(r"Peak pileup density ($\rm mm^{-1}$)", fontsize=22)
    plt.axvline(opt_fill_length, c="c", linestyle='--', lw=1)
    return fig, ax


'''
def cmp_pileup_density(results, myLumiModel, verbose=False, nb=None,
                                      ccrf=400e6, R12=46.66133302853659, CC_voltage=-3.4e6):

    sigma_z_list = np.linspace(-0.1, 0.1, 100)
    current_row = results.iloc[0]
    res = {}
    res["ip1"] = np.zeros((len(results),1,3))
    res["ip5"] = np.zeros((len(results),1,3))
    lambda_zeta = {}
    lambda_zeta["ip1"] = np.zeros((len(results), len(sigma_z_list)))
    lambda_zeta["ip5"] = np.zeros((len(results), len(sigma_z_list)))
    ips = [1,5]
    experiments = ["ATLAS", "CMS"]
    if myLumiModel.include_lhcb:
        res["ip8"] = np.zeros((len(results),1,3))
        lambda_zeta["ip8"] = np.zeros((len(results), len(sigma_z_list)))
        ips = [1,5,8]
        experiments = ["ATLAS", "CMS", "LHCb"]
    for counter, (ip, experiment) in enumerate(zip(ips, experiments)):
        for counter2, (index, current_row) in enumerate(results.iterrows()):
            for i in range(len(current_row["emitx_B1"])):
                if myLumiModel.include_cc and ip != 8:
                    if getattr(myLumiModel, f"{experiment}_crossing") == "H":
                            myR12, myV_x = R12, CC_voltage
                            myR34, myV_y = 0, 0

                    else:
                            myR34, myV_y = R12, CC_voltage
                            myR12, myV_x = 0, 0

                    if i== 0 and counter2 == 0:
                        print("CC included!")
                        print("experiment: ", experiment)
                        print("R12: ", myR12)
                        print("V_x: ", myV_x)
                        print("R34: ", myR34)
                        print("V_y: ", myV_y)
                                        
                else:
                    if i == 0 and counter2 == 0:
                        print("CC not included!")
                    myV_x, myV_y, myR12, myR34 = 0, 0, 0, 0
                if nb is None:
                    if ip==1 or ip==5:
                        nb = myLumiModel.number_of_bunches_coll_ip15
                    elif ip==8:
                        nb = myLumiModel.number_of_bunches_coll_ip8

                test = lumi_integrals.longitudinal_luminous_region(sigma_z_list,
                    f=11245.5, nb=nb, 
                    N1=current_row["inten_B1"][i], N2=current_row["inten_B1"][i], 
                    x_1=current_row[f"on_sep{ip}_x"], x_2=-current_row[f"on_sep{ip}_x"], 
                    y_1=current_row[f"on_sep{ip}_x"], y_2=-current_row[f"on_sep{ip}_x"], 
                    px_1=current_row[f"xing_ip{ip}_x"]/2., px_2=-current_row[f"xing_ip{ip}_x"]/2., 
                    py_1=current_row[f"xing_ip{ip}_y"]/2., py_2=-current_row[f"xing_ip{ip}_y"]/2., 
                    energy_tot1=myLumiModel.En*1e9, energy_tot2=myLumiModel.En*1e9, 
                    deltap_p0_1=0, deltap_p0_2=0, 
                    epsilon_x1=current_row["emitx_B1"][i], epsilon_x2=current_row["emitx_B2"][i], 
                    epsilon_y1=current_row["emity_B1"][i], epsilon_y2=current_row["emity_B2"][i], 
                    sigma_z1=current_row["bl_s_B1"][i]*clight/4., sigma_z2=current_row["bl_s_B2"][i]*clight/4., 
                    beta_x1=current_row[f"betax_ip{ip}"], beta_x2=current_row[f"betax_ip{ip}"], 
                    beta_y1=current_row[f"betay_ip{ip}"], beta_y2=current_row[f"betay_ip{ip}"], 
                    alpha_x1=0, alpha_x2=0, 
                    alpha_y1=0, alpha_y2=0, 
                    dx_1=0, dx_2=0, dy_1=0, dy_2=0, 
                    dpx_1=0, dpx_2=0, dpy_1=0, dpy_2=0, 
                    CC_V_x_1=getattr(myLumiModel,'cc_b1_fraction',1.0)*myV_x, CC_f_x_1=ccrf, CC_phase_x_1=0, 
                    CC_V_x_2=-getattr(myLumiModel,'cc_b2_fraction',1.0)*myV_x, CC_f_x_2=ccrf, CC_phase_x_2=0, 
                    CC_V_y_1=getattr(myLumiModel,'cc_b1_fraction',1.0)*myV_y, CC_f_y_1=ccrf, CC_phase_y_1=0, 
                    CC_V_y_2=-getattr(myLumiModel,'cc_b2_fraction',1.0)*myV_y, CC_f_y_2=ccrf, CC_phase_y_2=0, 
                    R12_1=myR12, R22_1=0, R34_1=myR34, R44_1=0, 
                    R12_2=myR12, R22_2=0, R34_2=myR34, R44_2=0, 
                    verbose=verbose, sigma_integration=3.0)
                #print(res[f"ip{ip}"])
                #print(myLumiModel.sigmaBOff_m2/myLumiModel.frev/nb )
                res[f"ip{ip}"][index, 0, 1] = np.max(myLumiModel.sigmaBOff_m2/myLumiModel.frev/nb * np.array(test[0]))
                lambda_zeta[f"ip{ip}"][index, :] = myLumiModel.sigmaBOff_m2/myLumiModel.frev/nb * np.array(test[0])

    results["pileup_density_ip5"] = np.mean(res["ip5"][:,:,1]*1e-3, axis=1)
    results["pileup_density_ip1"] = np.mean(res["ip1"][:,:,1]*1e-3, axis=1)
    results["pileup_density_avg"] = np.mean([np.mean(res["ip5"][:,:,1]*1e-3, axis=1), np.mean(res["ip1"][:,:,1]*1e-3, axis=1)], axis=0)
    if myLumiModel.include_lhcb:
        results["pileup_density_ip8"] = np.mean(res["ip8"][:,:,1]*1e-3, axis=1)
    return results, lambda_zeta

'''
