import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time as timer
import seaborn as sns
sns.set_style('whitegrid')
import xesmf as xe

import os
cwd = os.getcwd()

import sys


from utils_withmoist import odefunc_ebm_total_lat_firstordertaylor as odefunc_ebm_total_lat
from utils_withmoist import noise_mat_white
from utils_withmoist import odefunc_ebm_total_lat_variableD_plusterm_v3
from utils_withmoist import itoEuler_withnoisemat_explicit

# #Model parameters

D = 0.2598 # D = 0.2598 W/(m2 K) is the value used by TF10. D reported in pdf/cp
n = 51 #grid resolution
lat = np.linspace(-90,90,n) #DEFINED IN LATITUDE SPACE
dlat = 180/(n-1)
x = np.sin(np.deg2rad(lat))

# Moisture parameters
relhum = 0.8   # relative humidity
eps = 0.622    # moisture constant
psfc = 9.8e4   # (Pa)
e0 = 611.2     # vap. press (Pa)
a = 17.67
b = 243.5   # sat vap constants
L = 2.45e6         # latent heat of vaporization (J kg-1)
cp = 1004         # (J kg-1 K-1)


###########These don't get used in perturbation mode. But the way I've written/modified model code is a bit rigid and likes getting input for these fields.
A = 193 # OLR when T = 0 (W m^-2)
S0 = 420 # insolation at equator (W m^-2)
S2 = 240 # insolation spatial dependence (W m^-2)
a0 = 0.7
a2 = 0.1
ai = 0.4 
S = S0-S2*x**2 # insolation
aw = a0-a2*x**2 # open water albedo
drop_set=0
ramp_control=1
#######


def coarsen_numpy_single(arr, factor_time,factor_space):
    shape = arr.shape
    new_shape = (shape[0] // factor_time, factor_time, shape[1]//factor_space,factor_space)

    # Reshape and apply the mean operation along the first axis (after reshaping)
    coarsened_arr = arr.reshape(new_shape).mean(axis=(1,3))
    return coarsened_arr


temp_sat=xr.open_dataset('/glade/derecho/scratch/aurorabf/EBM/moist/test_variance_scaling/regridded_trefht_40members_CESM1.nc')
temp_sat=temp_sat.__xarray_dataarray_variable__
cesm_temps_mean=temp_sat.mean('ensemble_member')


ds_out = xr.Dataset(
    {
        "lat": (["lat"],np.linspace(-90,90,180), {"units": "degrees_north"}),
        "lon": (["lon"], np.linspace(-180,180,180), {"units": "degrees_east"}),
    }
)
ds_out

ice=xr.open_dataset('/glade/derecho/scratch/aurorabf/EBM/moist/test_variance_scaling/regridded_ifrac_40members_CESM1.nc').__xarray_dataarray_variable__


regridder=xe.Regridder(ice,ds_out, method="nearest_s2d", periodic=True, reuse_weights=False,ignore_degenerate=True)
regridded_ice=regridder(ice)

landmask = xr.where(regridded_ice[0,0,:,:].isnull(), 1, 0)


is_polar_land = (landmask == 1) & (
    (landmask.lat < -55) | #checked and this is the same if i had done -60 instead
    ((landmask.lat > 68) & (landmask.lon > -72) & (landmask.lon < -15)) | #upper part of greenland
    ((landmask.lat < 68) & (landmask.lat > 60) & (landmask.lon > -55) & (landmask.lon < -28))#lower part of greenland
)

# Force ice to 1.0 only on those specific polar land pixels
# This makes the heatcap to use ice_heatcap instead of land_heatcap there
ice_adjusted = xr.where(is_polar_land, 1.0, regridded_ice)


def heatcap(cw,ice,factor,landmask):
    ocean_heatcap=cw
    ice_heatcap=cw/factor
    land_heatcap=cw/600
    heatcap=((ice_heatcap*ice.fillna(0)) + (ocean_heatcap*(1-ice.fillna(0))*(1-landmask))+land_heatcap*landmask*(1-is_polar_land))
    return heatcap


lambda_local=-xr.open_dataset("/glade/u/home/aurorabf/SMEBM/data/variability_feedback_2000to2025_sst_oct2026.nc").slope #we checked this is exactly the same as "/glade/derecho/scratch/aurorabf/EBM/moist/test_variance_scaling/variability_feedback_2000to2025_sst.nc"
lambda_local2=-xr.open_dataset("/glade/u/home/aurorabf/SMEBM/data/variability_feedback_2075to2100_sst_oct2026.nc").slope



ds_target = xr.Dataset({
    "lat": (["lat"], lat),
    "lon": (["lon"], np.linspace(-180,180,180)), # Dummy lon because xesmf needs lon
})
regridder=xe.Regridder(lambda_local.expand_dims(lon=np.linspace(-180,180,180)),ds_target, method='nearest_s2d',ignore_degenerate=True)

def regrid(variable,regridder,expand_lon=1):
    if expand_lon==1:
        variable_expanded=variable.expand_dims(lon=np.linspace(-180,180,180))
        variable_regridded = regridder(variable_expanded)
        variable_final = variable_regridded.sel(lon=-180)
    else:
        variable_final=regridder(variable)
    return variable_final

lam1=np.array(regrid(lambda_local,regridder))
lam2=np.array(regrid(lambda_local2,regridder))
cw_alltime=np.array(regrid(heatcap(1.9*4.6,ice_adjusted.mean('ensemble_member'),65,landmask),regridder,expand_lon=0).mean('lon'))
cesm_temps_xr=regrid(cesm_temps_mean,regridder,expand_lon=0).mean('lon')-273.15



########
def run_control(time_rcp,i):
    lam=lam1
    cw=cw_alltime[12*80:12*105,:].mean(axis=0)
    target_profile=np.array(cesm_temps_xr[12*80:12*105,:].mean(axis=0))
    
    odefunc_control=lambda T,t:odefunc_ebm_total_lat(T,t,aw,ai,S,A,lam,D,dlat,lat,cw,ramp_on=ramp_control,F=0,drop=drop_set,relhum=relhum,eps=eps,psfc=psfc,e0=e0,a=a,b=b,L=L,cp=cp,T_init=target_profile)
    
    
    dt=np.diff(time_rcp)[0]
    
    
    
    results = {}
    t0 = timer.time()
    dt = np.diff(time_rcp)[0]
    
    
    sol_variability = itoEuler_withnoisemat_explicit(
        odefunc_control,
        noise_mat_white(i, int(np.size(time_rcp)), cw, noise),
        np.zeros(np.size(target_profile)),
        tspan=time_rcp, dx=dx
    )

    coarsened_var=coarsen_numpy_single(sol_variability,int(np.size(time_rcp)/time_rcp[-1]),1)
    sol_variability_xr=xr.DataArray(coarsened_var,dims=["time","lat"],coords={"time": np.linspace(1,int(time_rcp[-1]),int(time_rcp[-1])),"lat":lat,"timestep":np.diff(time_rcp)[0]})
    return sol_variability_xr


def run_changetemp(time_rcp,i):
    lam=lam1
    cw=cw_alltime[12*80:12*105,:].mean(axis=0)
    target_profile=np.array(cesm_temps_xr[-12*25:,:].mean(axis=0))
    
    odefunc_control=lambda T,t:odefunc_ebm_total_lat(T,t,aw,ai,S,A,lam,D,dlat,lat,cw,ramp_on=ramp_control,F=0,drop=drop_set,relhum=relhum,eps=eps,psfc=psfc,e0=e0,a=a,b=b,L=L,cp=cp,T_init=target_profile)
    
    
    dt=np.diff(time_rcp)[0]
    
    
    
    results = {}
    t0 = timer.time()
    dt = np.diff(time_rcp)[0]
    
    
    sol_variability = itoEuler_withnoisemat_explicit(
        odefunc_control,
        noise_mat_white(i, int(np.size(time_rcp)), cw, noise),
        np.zeros(np.size(target_profile)),
        tspan=time_rcp, dx=dx
    )

    coarsened_var=coarsen_numpy_single(sol_variability,int(np.size(time_rcp)/time_rcp[-1]),1)
    sol_variability_xr=xr.DataArray(coarsened_var,dims=["time","lat"],coords={"time": np.linspace(1,int(time_rcp[-1]),int(time_rcp[-1])),"lat":lat,"timestep":np.diff(time_rcp)[0]})
    return sol_variability_xr


def run_forced(time_rcp,i):
    lam=lam2
    cw=cw_alltime[-12*25:,:].mean(axis=0)
    target_profile=np.array(cesm_temps_xr[-12*25:,:].mean(axis=0))
    
    odefunc_control=lambda T,t:odefunc_ebm_total_lat(T,t,aw,ai,S,A,lam,D,dlat,lat,cw,ramp_on=ramp_control,F=0,drop=drop_set,relhum=relhum,eps=eps,psfc=psfc,e0=e0,a=a,b=b,L=L,cp=cp,T_init=target_profile)

    
    
    dt=np.diff(time_rcp)[0]
    
    
    
    results = {}
    t0 = timer.time()
    dt = np.diff(time_rcp)[0]
    
    
    sol_forced_variability = itoEuler_withnoisemat_explicit(
        odefunc_control,
        noise_mat_white(i, int(np.size(time_rcp)), cw, noise),
        np.zeros(np.size(target_profile)),
        tspan=time_rcp, dx=dx
    )
    coarsened_var=coarsen_numpy_single(sol_forced_variability,int(np.size(time_rcp)/time_rcp[-1]),1)
    
    sol_variability_xr=xr.DataArray(coarsened_var,dims=["time","lat"],coords={"time": np.linspace(1,int(time_rcp[-1]),int(time_rcp[-1])),"lat":lat,"timestep":np.diff(time_rcp)[0]})
    return sol_variability_xr


def run_forced_tempandice(time_rcp,i):
    lam=lam1
    cw=cw_alltime[-12*25:,:].mean(axis=0)
    target_profile=np.array(cesm_temps_xr[-12*25:,:].mean(axis=0))
    
    odefunc_control=lambda T,t:odefunc_ebm_total_lat(T,t,aw,ai,S,A,lam,D,dlat,lat,cw,ramp_on=ramp_control,F=0,drop=drop_set,relhum=relhum,eps=eps,psfc=psfc,e0=e0,a=a,b=b,L=L,cp=cp,T_init=target_profile)

    
    
    dt=np.diff(time_rcp)[0]
    
    
    
    results = {}
    t0 = timer.time()
    dt = np.diff(time_rcp)[0]
    
    
    sol_forced_variability = itoEuler_withnoisemat_explicit(
        odefunc_control,
        noise_mat_white(i, int(np.size(time_rcp)), cw, noise),
        np.zeros(np.size(target_profile)),
        tspan=time_rcp, dx=dx
    )
    coarsened_var=coarsen_numpy_single(sol_forced_variability,int(np.size(time_rcp)/time_rcp[-1]),1)
    
    sol_variability_xr=xr.DataArray(coarsened_var,dims=["time","lat"],coords={"time": np.linspace(1,int(time_rcp[-1]),int(time_rcp[-1])),"lat":lat,"timestep":np.diff(time_rcp)[0]})
    return sol_variability_xr



# ##

def dh_func(T): #note this has a times cp that we dont use when we use h in diffusion because that's in the D
    relhum = 0.8   # relative humidity
    eps = 0.622    # moisture constant
    psfc = 9.8e4   # (Pa)
    e0 = 611.2     # vap. press (Pa)
    a = 17.67
    b = 243.5   # sat vap constants
    L = 2.45e6         # latent heat of vaporization (J kg-1)
    cp = 1004         # (J kg-1 K-1)
    dq=eps*relhum/psfc*e0*((a*b)/(b+T)**2)*np.exp(a*(T)/(b+(T)))
    theta_e = cp+ L*dq #this just got updated. april 23
    return theta_e


def run_term_a(target_profile,time_rcp,i):
    lam=lam1
    cw=cw_alltime[12*80:12*105,:].mean(axis=0)
    
    odefunc_control=lambda T,t:odefunc_ebm_total_lat_variableD_plusterm_v3(T,t,aw,ai,S,A,lam,D,dlat,lat,cw,ramp_on=ramp_control,F=0,drop=drop_set,relhum=0,eps=eps,psfc=psfc,e0=e0,a=a,b=b,L=L,cp=cp,T_init=target_profile,dhdT=dh_func(target_profile)/cp,last_term=0)
    
    
    dt=np.diff(time_rcp)[0]
    
    
    
    results = {}
    t0 = timer.time()
    dt = np.diff(time_rcp)[0]
    
    
    sol_variability = itoEuler_withnoisemat_explicit(
        odefunc_control,
        noise_mat_white(i, int(np.size(time_rcp)), cw, noise),
        np.zeros(np.size(target_profile)),
        tspan=time_rcp, dx=dx
    )

    coarsened_var=coarsen_numpy_single(sol_variability,int(np.size(time_rcp)/time_rcp[-1]),1)
    sol_variability_xr=xr.DataArray(coarsened_var,dims=["time","lat"],coords={"time": np.linspace(1,int(time_rcp[-1]),int(time_rcp[-1])),"lat":lat,"timestep":np.diff(time_rcp)[0]})
    return sol_variability_xr

def run_term_b(target_profile,time_rcp,i):
    lam=lam1
    cw=cw_alltime[12*80:12*105,:].mean(axis=0)
    
    odefunc_control=lambda T,t:odefunc_ebm_total_lat_variableD_plusterm_v3(T,t,aw,ai,S,A,lam,D,dlat,lat,cw,ramp_on=ramp_control,F=0,drop=drop_set,relhum=0,eps=eps,psfc=psfc,e0=e0,a=a,b=b,L=L,cp=cp,T_init=target_profile,dhdT=dh_func(target_profile)/cp,last_term=0,include_b=1)
    
    
    dt=np.diff(time_rcp)[0]
    
    
    
    results = {}
    t0 = timer.time()
    dt = np.diff(time_rcp)[0]
    
    
    sol_variability = itoEuler_withnoisemat_explicit(
        odefunc_control,
        noise_mat_white(i, int(np.size(time_rcp)), cw, noise),
        np.zeros(np.size(target_profile)),
        tspan=time_rcp, dx=dx
    )

    coarsened_var=coarsen_numpy_single(sol_variability,int(np.size(time_rcp)/time_rcp[-1]),1)
    sol_variability_xr=xr.DataArray(coarsened_var,dims=["time","lat"],coords={"time": np.linspace(1,int(time_rcp[-1]),int(time_rcp[-1])),"lat":lat,"timestep":np.diff(time_rcp)[0]})
    return sol_variability_xr


def run_term_c(target_profile,time_rcp,i):
    lam=lam1
    cw=cw_alltime[12*80:12*105,:].mean(axis=0)
    
    odefunc_control=lambda T,t:odefunc_ebm_total_lat_variableD_plusterm_v3(T,t,aw,ai,S,A,lam,D,dlat,lat,cw,ramp_on=ramp_control,F=0,drop=drop_set,relhum=0,eps=eps,psfc=psfc,e0=e0,a=a,b=b,L=L,cp=cp,T_init=target_profile,dhdT=dh_func(target_profile)/cp,last_term=1)
    
    
    dt=np.diff(time_rcp)[0]
    
    
    
    results = {}
    t0 = timer.time()
    dt = np.diff(time_rcp)[0]
    
    
    sol_variability = itoEuler_withnoisemat_explicit(
        odefunc_control,
        noise_mat_white(i, int(np.size(time_rcp)), cw, noise),
        np.zeros(np.size(target_profile)),
        tspan=time_rcp, dx=dx
    )

    coarsened_var=coarsen_numpy_single(sol_variability,int(np.size(time_rcp)/time_rcp[-1]),1)
    sol_variability_xr=xr.DataArray(coarsened_var,dims=["time","lat"],coords={"time": np.linspace(1,int(time_rcp[-1]),int(time_rcp[-1])),"lat":lat,"timestep":np.diff(time_rcp)[0]})
    return sol_variability_xr


####
noise=10*np.ones(np.size(lat))
dx=np.diff(lat)[0]
ensemble_size = 100



num_time=50000*1.2*16
num_years=150
time_rcp=np.linspace(0,num_years,int(num_time))
control_members = []
temp_members = []
temp_and_ice_members= []
forced_members = []



for i in range(ensemble_size):
    sol_1 = run_control(time_rcp, i)
    sol_temp=run_changetemp(time_rcp,i)
    sol_forced = run_forced(time_rcp, i)
    sol_tempandice=run_forced_tempandice(time_rcp,i)

    

    control_members.append(sol_1.assign_coords(ensemble=i))
    temp_members.append(sol_temp.assign_coords(ensemble=i))
    temp_and_ice_members.append(sol_tempandice.assign_coords(ensemble=i))
    forced_members.append(sol_forced.assign_coords(ensemble=i))

    print(i)

sol_control = xr.concat(control_members, dim='ensemble_member')
sol_temp = xr.concat(temp_members, dim='ensemble_member')
sol_tempandice=xr.concat(temp_and_ice_members, dim='ensemble_member')
sol_forced=xr.concat(forced_members, dim='ensemble_member')

sol_control.to_netcdf("/glade/derecho/scratch/aurorabf/EBM/moist/test_variance_scaling/tune_timestep_cesm1_variability_control_cw8pt74_ratio65land600_ensemble_longrun_formultilevel_forcomparisonwithlevelsmay_notikhonov_sst_quarterdt.nc")
sol_temp.to_netcdf("/glade/derecho/scratch/aurorabf/EBM/moist/test_variance_scaling/tune_timestep_cesm1_variability_temponly_cw8pt74_ratio65land600_ensemble_longrun_formultilevel_forcomparisonwithlevelsmay_notikhonov_sst_quarterdt.nc")
sol_tempandice.to_netcdf("/glade/derecho/scratch/aurorabf/EBM/moist/test_variance_scaling/tune_timestep_cesm1_variability_tempandice_cw8pt74_ratio65land600_ensemble_longrun_formultilevel_forcomparisonwithlevelsmay_notikhonov_sst_quarterdt.nc")
sol_forced.to_netcdf("/glade/derecho/scratch/aurorabf/EBM/moist/test_variance_scaling/tune_timestep_cesm1_variability_forced_cw8pt74_ratio65land600_ensemble_longrun_formultilevel_forcomparisonwithlevelsmay_notikhonov_sst_quarterdt.nc")

# ###
control_terma_members = []
control_termb_members = []
control_termc_members = []

forced_terma_members = []
forced_termb_members = []
forced_termc_members = []


for i in range(ensemble_size):
    forced_sol_terma = run_term_a(np.array(cesm_temps_xr[-12*25:,:].mean(axis=0)),time_rcp, i)
    forced_sol_termb = run_term_b(np.array(cesm_temps_xr[-12*25:,:].mean(axis=0)),time_rcp, i)
    forced_sol_termc = run_term_c(np.array(cesm_temps_xr[-12*25:,:].mean(axis=0)),time_rcp, i)

    forced_terma_members.append(forced_sol_terma.assign_coords(ensemble=i))
    forced_termb_members.append(forced_sol_termb.assign_coords(ensemble=i))
    forced_termc_members.append(forced_sol_termc.assign_coords(ensemble=i))

    control_sol_terma = run_term_a(np.array(cesm_temps_xr[12*80:12*105,:].mean(axis=0)),time_rcp, i)
    control_sol_termb = run_term_b(np.array(cesm_temps_xr[12*80:12*105,:].mean(axis=0)),time_rcp, i)
    control_sol_termc = run_term_c(np.array(cesm_temps_xr[12*80:12*105,:].mean(axis=0)),time_rcp, i)

    control_terma_members.append(control_sol_terma.assign_coords(ensemble=i))
    control_termb_members.append(control_sol_termb.assign_coords(ensemble=i))
    control_termc_members.append(control_sol_termc.assign_coords(ensemble=i))

    print(i)

forced_sol_terma = xr.concat(forced_terma_members, dim='ensemble_member')
forced_sol_termb = xr.concat(forced_termb_members, dim='ensemble_member')
forced_sol_termc = xr.concat(forced_termc_members, dim='ensemble_member')
control_sol_terma = xr.concat(control_terma_members, dim='ensemble_member')
control_sol_termb = xr.concat(control_termb_members, dim='ensemble_member')
control_sol_termc = xr.concat(control_termc_members, dim='ensemble_member')

forced_breakdown=xr.concat([forced_sol_terma,forced_sol_termb,forced_sol_termc],dim='term')
control_breakdown=xr.concat([control_sol_terma,control_sol_termb,control_sol_termc],dim='term')


control_breakdown.to_netcdf("/glade/derecho/scratch/aurorabf/EBM/moist/test_variance_scaling/tune_timestep_cesm1_variability_controldecomposition_cw8pt74_ratio65land600_ensemble_longrun_formultilevel_fixmay_notikhonov_sst_quarterdt.nc")
forced_breakdown.to_netcdf("/glade/derecho/scratch/aurorabf/EBM/moist/test_variance_scaling/tune_timestep_cesm1_variability_forceddecomposition_cw8pt74_ratio65land600_ensemble_longrun_formultilevel_fixmay_notikhonov_sst_quarterdt.nc")
