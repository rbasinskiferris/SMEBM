import scipy.io
import netCDF4
import xarray as xr
import numpy as np
import xesmf as xe

#TEMP
temp = [] 
for time_period in range(0, 3):
    ensemble_members = []  
    for i in range(1, 34):  
        if time_period == 0: #1920 to 2005
            if i==1:
                f="/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.B20TRC5CNBDRD.f09_g16.00%d.cam.h0.TS.185001-200512.nc" % i
            elif 1 < i < 10:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.B20TRC5CNBDRD.f09_g16.00%d.cam.h0.TS.192001-200512.nc" % i
            else:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.B20TRC5CNBDRD.f09_g16.0%d.cam.h0.TS.192001-200512.nc" % i
        if time_period ==1: #2005 to 2080
            if i < 10:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.BRCP85C5CNBDRD.f09_g16.00%d.cam.h0.TS.200601-208012.nc" % i
            else:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.BRCP85C5CNBDRD.f09_g16.0%d.cam.h0.TS.200601-208012.nc" % i
        if time_period ==2: #2080 to 2100
            if i < 10:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.BRCP85C5CNBDRD.f09_g16.00%d.cam.h0.TS.208101-210012.nc" % i
            else:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.BRCP85C5CNBDRD.f09_g16.0%d.cam.h0.TS.208101-210012.nc" % i
        s = xr.open_dataset(f)['TS']
        if i==1 and time_period==0:
            s=s[70*12:]
        s = s.assign_coords(ensemble_member=i)  
        ensemble_members.append(s)  
    combined_data = xr.concat(ensemble_members, dim="ensemble_member")
    temp.append(combined_data) 
final_data_first33_ts = xr.concat(temp, dim="time")

temp = [] 
for time_period in range(0, 2):
    ensemble_members = []  
    for i in range(101, 108):  
        if time_period == 0:  # 1920 to 2005
            if i == 106:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.B20TRC5CNBDRD.f09_g16.034.cam.h0.TS.192001-200512.nc"
            if i == 107:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.B20TRC5CNBDRD.f09_g16.035.cam.h0.TS.192001-200512.nc"
            elif i < 106:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.B20TRC5CNBDRD.f09_g16.%d.cam.h0.TS.192001-200512.nc" % i
        if time_period == 1:  # 2005 to 2080
            if i == 106:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.BRCP85C5CNBDRD.f09_g16.034.cam.h0.TS.200601-210012.nc"
            if i == 107:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.BRCP85C5CNBDRD.f09_g16.035.cam.h0.TS.200601-210012.nc"
            elif i < 106:
                f = "/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.BRCP85C5CNBDRD.f09_g16.%d.cam.h0.TS.200601-210012.nc" % i
        
        # Open dataset and get temperature data
        s = xr.open_dataset(f)['TS']
        
        # Assign ensemble_member coordinate
        s = s.assign_coords(ensemble_member=i-101+34)

        # Interpolate if i is 106 or 107 to match latitudes of 105
        if i < 106:
            #plt.plot(np.array(ensemble_members[0].lat)-np.array(s.lat)) #compare this which is slightly different from 0
            #plt.plot(np.array(ensemble_members[0].lat)-np.array(ensemble_members[1].lat)) #to this which is exactly 0 (for the first few members)
            reference_lat = final_data_first33_ts.lat  
            s = s.interp(lat=reference_lat)  # Interpolate to match latitude values of i=105. this has to be done because the latitudes are off by 1E-15 (to see this, get rid of this interpolation and uncomment the plot in the line above)
        ensemble_members.append(s)  
    combined_data = xr.concat(ensemble_members, dim="ensemble_member")
    temp.append(combined_data) 
final_data_next7_ts = xr.concat(temp, dim="time")

final_data=xr.concat([final_data_first33_ts,final_data_next7_ts],dim="ensemble_member")


f='/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/TS/b.e11.B20TRC5CNBDRD.f09_g16.035.cam.h0.TS.192001-200512.nc'
file2read = netCDF4.Dataset(f,'r')
#print(file2read.variables)
lat = xr.open_dataset(f)['lat']
lon = xr.open_dataset(f)['lon']
time=xr.open_dataset(f)['time']
temp2=xr.open_dataset(f)['TS']


ds_out = xr.Dataset(
    {
        "lat": (["lat"],np.linspace(-90,90,180), {"units": "degrees_north"}),
        "lon": (["lon"], np.linspace(-180,180,180), {"units": "degrees_east"}),
    }
)
ds_out

regridder_forcesm = xe.Regridder(temp2[:,:,:], ds_out, "bilinear",periodic=True) #this is now the regridder object that can be applied to any of the original CESM fields
regrid_cesm = regridder_forcesm(temp2[:,:,:])


regridded_temp=[]
for i in range(0,40): #(0,10):
    s = regridder_forcesm(final_data[i,:,:,:])
    regridded_temp.append(s)
    print(i)
regridded_temp=xr.concat(regridded_temp,dim='ensemble_member')


print('regridded')
regridded_temp.to_netcdf('/glade/derecho/scratch/aurorabf/EBM/moist/test_variance_scaling/regridded_temp_40members_CESM1.nc')

del regridded_temp
del final_data