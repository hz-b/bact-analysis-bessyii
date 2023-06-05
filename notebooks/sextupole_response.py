#!/usr/bin/env python
# coding: utf-8

# In[1]:


from databroker import catalog
from databroker.queries import TimeRange
import datetime
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import tqdm
import pandas as pd
import xarray as xr
import numpy as np


# In[3]:


from bact_math_utils.misc import enumerate_changed_value, CountSame, EnumerateUniqueJustSeen
from bact_math_utils.linear_fit import linear_fit_1d
from bact_math_utils.tune import to_momentum, tune_frequency_to_fractional_tune


# In[ ]:


db = catalog['heavy_local']


# # Sextupole response evaluated over chromaticity

# Evaluation:
# 
# * Read measured data (see http://elog-v2.trs.bessy.de:8080/Machine+Devel.,+Comm./2032 for details on the measurement)
# * Sort data to the different magnets
# * Show  / fit tune shift versus momentum shift

# ## Select data

# Just to show how to find the data

# In[ ]:


t_search =  db.search(TimeRange(since="2022-11-2", until="2022-11-25"))
possible_sextupole_response = t_search.search(dict(nickname="sextupole response"))

l = []
for uid in possible_sextupole_response:
    run = db[uid]
    start = run.metadata["start"]    
    ts_start = datetime.datetime.fromtimestamp(int(start['time']))
    stop = run.metadata["stop"]
    
    if not stop:
        print(f'{uid} {ts_start} ----')
        continue
    ts_end = datetime.datetime.fromtimestamp(int(stop['time']))
    #print(f'{uid}  {ts_start} {ts_end}')./lib/python3.9/site-packages/bluesky/callbacks/core.py
    print(f'<tr><td>{uid}</td><td> {ts_start} </td><td>{ts_end}</td></tr>')


# The following uid is the one to be used for most magnet data

# In[ ]:


uid = "eb42c33c-a218-4fb1-a1c2-bd80403e2cc9"


# In[ ]:


run = db[uid]
run


# ### Read data

# First step requires roughly 1.5 minutes

# In[ ]:


start = datetime.datetime.now()
data_ = run.primary.to_dask()
end = datetime.datetime.now()
f"Reading data as dask arrays required:{end - start}"


# In[ ]:


data_


# In[ ]:


variables = ["cs_" + var_name for var_name in ["tm_vert_readback", "tm_hor_readback", "mc_setpoint", "nu_x_nu"]]
variables


# Now loading all dask array to memory requires roughtly 20 seconds

# In[ ]:


start = datetime.datetime.now()
for name, item in tqdm.tqdm(data_.items(),  total=len(data_.variables)):
    item.load()
end = datetime.datetime.now()
f"Loading dask array data to memory required:{end - start}"


# In[ ]:


# data_.load()


# In[ ]:


data_;


# ### Reducing data to the last ones

# While the phyical process is rather fast, the spectrum analyser needs a bit of time to update the wole measurement.
# Therefore lets ignore the first one s
# 
# These are rather the good ones, still jitter at the beginng

# In[ ]:


data = data_.isel(time=data_.cs_cs_setpoint > 3)


# Investigate the current in the machine 

# In[ ]:


data.cs_topup_current.plot()


# In[ ]:


magnet_names = [val[1] for val in CountSame()(data.sc_sel_selected.values) if val[0] == 0]


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=[8, 6])
ax.plot(
    data.sc_sel_selected.values,
    data.cs_topup_current
)
ax.set_ylabel("I [mA]")
plt.setp(ax.yaxis.label, "fontsize", "x-large")
for tick in ax.xaxis.get_majorticklabels() + ax.yaxis.get_majorticklabels():
    tick.set_fontsize("x-large")
    
for tick in ax.xaxis.get_majorticklabels():
    tick.set_verticalalignment("top")
    tick.set_horizontalalignment("right")
    tick.set_rotation(45)    


# In[ ]:





# ## Data evaluation

# ### Test for one magnet for for one current

# How the power converters were executed

# In[ ]:


import bact_math_utils.misc


# In[ ]:


selected_power_converter = "s3ptr"


# In[ ]:


sel = data.isel(time=data.sc_sel_selected == selected_power_converter)


# In[ ]:


sel.time.shape


# In[ ]:


sel.cs_mc_frequency_readback / sel.cs_md_rv_freq;


# In[ ]:


def calculate_revolution_frequency(freq: float, n_bunches:  int) -> float:
    return freq / n_bunches


# In[ ]:


desc, = run.primary.metadata["descriptors"]


# In[ ]:


config_data = desc["configuration"]["cs"]["data"]


# In[ ]:


alpha = config_data["cs_md_alpha"]
n_bunches = config_data["cs_md_n_bunches"]
ref_freq = config_data["cs_ref_freq"]
rv_freq =  ref_freq / n_bunches
ref_freq, rv_freq


# In[ ]:


momentum = to_momentum(sel.cs_mc_frequency_readback, ref_freq=ref_freq, alpha=alpha)


# In[ ]:


frac_tune_x = tune_frequency_to_fractional_tune(sel.cs_tm_bpm_hor_readback, ring_frequency=rv_freq)
frac_tune_y = tune_frequency_to_fractional_tune(sel.cs_tm_bpm_vert_readback, ring_frequency=rv_freq)


# In[ ]:


sel.cs_tm_bpm_hor_readback;


# In[ ]:


plt.plot(momentum, frac_tune_y)


# Plot it separately for each run 

# Use the current to index the run 

# In[ ]:


changed_value = enumerate_changed_value(sel.sc_sel_r_setpoint)

fig, axes = plt.subplots(1, 2, figsize=[16, 12])
ax_x, ax_y = axes
colors = []
for cnt in np.unique(changed_value):
    idx = changed_value == cnt
    line, = ax_x.plot(momentum[idx], frac_tune_x[idx], '.-', linewidth=.5)
    colors.append(line.get_color())
    ax_y.plot(momentum[idx], frac_tune_y[idx], '.-', linewidth=.5, color=line.get_color())
    
ax_x.set_xlabel("$\Delta p/p")
ax_y.set_xlabel("$\Delta p/p")
ax_x.set_ylabel("$Q_x$")
ax_y.set_ylabel("$Q_y$")    


# In[ ]:


changed_value = enumerate_changed_value(sel.sc_sel_r_setpoint)
cv_idx = np.unique(changed_value)


fig, axes = plt.subplots(1, 2, figsize=[16, 12])
ax_x, ax_y = axes

x_coeff_avg =  np.polyfit(momentum, frac_tune_x, deg = 1)
y_coeff_avg =  np.polyfit(momentum, frac_tune_y, deg = 1)


l = []
for cnt, color in zip(cv_idx, colors):
    idx = changed_value == cnt
    mms = momentum[idx]
    
    ftx = frac_tune_x[idx]
    fty = frac_tune_y[idx]
    
    dI = sel.sc_sel_p_setpoint[idx].mean()
    p_x, dp_x = linear_fit_1d(mms, ftx)
    p_y, dp_y = linear_fit_1d(mms, fty)
    
    l.append(np.array([(p_x, dp_x),(p_y, dp_y)]))
    exp_frac_tune_x = np.polyval(x_coeff_avg, mms)
    exp_frac_tune_y = np.polyval(y_coeff_avg, mms)

    mmse = np.linspace(mms.min(), mms.max())
    eftxr = np.polyval(x_coeff_avg, mmse)
    eftyr = np.polyval(y_coeff_avg, mmse)
    eftx = np.polyval(p_x, mmse)
    efty = np.polyval(p_y, mmse)
    
    s = 1000
    ax_x.plot(mms , (ftx - exp_frac_tune_x) * s, '.:', linewidth=.25, color=color)
    ax_y.plot(mms , (fty - exp_frac_tune_y) * s, '.:', linewidth=.25, color=color)
    ax_x.plot(mmse, (eftx - eftxr) * s, '-', linewidth=.25, color=color)
    ax_y.plot(mmse, (efty - eftyr) * s, '-', linewidth=.25, color=color)
    
ax_x.set_xlabel("$\Delta p/p$")
ax_y.set_xlabel("$\Delta p/p$")
ax_x.set_ylabel("$\Delta Q_x * 1000$")
ax_y.set_ylabel("$\Delta Q_y * 1000$")


# ## Tune shifts for all magnets and current indices

# ### Calculating tune shift for all magnets

# In[ ]:


def tune_shift_vs_excitation(sel, *, name):
    changed_value = enumerate_changed_value(sel.sc_sel_r_setpoint)
    cv_idx = np.unique(changed_value)
    
    momentum = to_momentum(sel.cs_mc_frequency_readback, ref_freq=ref_freq, alpha=alpha)
    frac_tune_x = tune_frequency_to_fractional_tune(sel.cs_tm_bpm_hor_readback, ring_frequency=rv_freq)
    frac_tune_y = tune_frequency_to_fractional_tune(sel.cs_tm_bpm_vert_readback, ring_frequency=rv_freq)
    
    l = []
    currents = []
    for cnt in cv_idx:
        idx = changed_value == cnt
        mms = momentum[idx]
    
        ftx = frac_tune_x[idx]
        fty = frac_tune_y[idx]
    
        dI = sel.sc_sel_r_setpoint[idx].mean()
        p_x, dp_x = linear_fit_1d(mms, ftx)
        p_y, dp_y = linear_fit_1d(mms, fty)
        l.append(np.array([(p_x, dp_x),(p_y, dp_y)]))
        currents.append(dI)
        
    da = xr.DataArray(data=np.array(l)[np.newaxis,:],
        dims=["name", "current_index", "plane",  "prop", "param",], 
        coords=[[name], cv_idx, ["x", "y"],  ["val", "err"], ["slope", "intercept"],]
    )

    da_cur = xr.DataArray(data=np.array(currents)[np.newaxis,:],
        dims=["name", "current_index"], 
        coords=[[name], cv_idx]
    )
    return da, da_cur


# In[ ]:


tmp = [tune_shift_vs_excitation(data.isel(time=data.sc_sel_selected == pc_name), name=pc_name) for pc_name in magnet_names]


# In[ ]:


current_shifts_ = xr.concat([t[1] for t in tmp], dim="name")
current_shifts_;


# In[ ]:


current_shifts = current_shifts_ - current_shifts_.sel(current_index=[0,3]).mean(axis=-1)
current_shifts


# In[ ]:


tune_shifts = xr.concat([t[0] for t in tmp], dim="name")
tune_shifts;


# In[ ]:


ds = xr.Dataset(dict(tunes=tune_shifts, currents=current_shifts))
ds.to_netcdf("chromaticity_vs_sextupole_current_fits.nc")


# ### Simple plots for one power converter

# In[ ]:


tune_shifts.coords["name"]


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=[24, 16])
ax_i, ax_s = axes
ax_i_x, ax_i_y = ax_i
ax_s_x, ax_s_y = ax_s


for pc_name in ["s4p2t6r", "s3p2t6r", "s4p1t6r", "s3p1t6r", "s4pd1r", "s3pd1r", "s4ptr", "s4pdr", "s3ptr", "s3pdr"]:
    ts4m = tune_shifts.sel(name=pc_name)
    currents = current_shifts.sel(name=pc_name)
    ax_i_x.errorbar(currents, ts4m.sel(plane="x", param="intercept", prop="val"), yerr=ts4m.sel(plane="x", param="intercept", prop="err"), label=pc_name)
    ax_i_y.errorbar(currents, ts4m.sel(plane="y", param="intercept", prop="val"), yerr=ts4m.sel(plane="y", param="intercept", prop="err"), label=pc_name)

    ax_s_x.errorbar(currents, ts4m.sel(plane="x", param="slope", prop="val"), yerr=ts4m.sel(plane="x", param="slope", prop="err"), label=pc_name)
    ax_s_y.errorbar(currents, ts4m.sel(plane="y", param="slope", prop="val"), yerr=ts4m.sel(plane="y", param="slope", prop="err"), label=pc_name)

ax_i_x.set_ylabel(r"$\Delta\nu_x$")
ax_i_y.set_ylabel(r"$\Delta\nu_y$")

ax_s_x.set_ylabel(r"$\Delta\nu_x / (\Delta P /  P) $")
ax_s_y.set_ylabel(r"$\Delta\nu_y / (\Delta P  / P) $")

ax_s_x.set_xlabel(r"$\Delta I [A]$")
ax_s_y.set_xlabel(r"$\Delta I [A]$")

ax_s_x.legend()

for ax in ax_i_x, ax_i_y, ax_s_x, ax_s_y:
    for tick in ax.xaxis.get_majorticklabels() + ax.yaxis.get_majorticklabels():
        tick.set_fontsize("x-large")
    ax.yaxis.label.set_fontsize("x-large")
ax_s_x.yaxis.label.set_fontsize("x-large")
ax_s_y.yaxis.label.set_fontsize("x-large")

