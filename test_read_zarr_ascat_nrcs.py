import xarray as xr 
import numpy as np 
from fibgrid.realization import FibGrid
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import matplotlib.pyplot as plt

lomax = 17
lomin = 15
lamax = 48
lamin = 47

ascat_ds = xr.open_zarr("/home/mpanfilo/radar01/RADAR/ascat/ascat_resample/ascat_METOPABC_12.5_ts_v2.zarr")
xr.set_options(display_expand_data=True)
data_vars_ascat = ascat_ds.data_vars
print(ascat_ds)

gpi_ = 2497479
swath = ascat_ds['swath_indicator'].isel(gpi=gpi_).values
pass_as_des = ascat_ds['as_des_pass'].isel(gpi=gpi_).values

mask = np.logical_and(swath==1, pass_as_des==1)

inc_angles = ascat_ds['incidence_angle'].isel(gpi=2497479, beam=2).values[mask]
print(inc_angles[0])

#time_stamps = ascat_ds['time'].isel(gpi=1650001).values
#print(time_stamps.max())

plt.hist(inc_angles[:1000], 100)
plt.title('first 1000 samples, swath=1, pass=1, beam=2')
plt.xlabel('incidence angle [deg]')
plt.savefig('inc_beam0_4.png')
plt.close()


'''
# here should be the xarray, not numpy array. Do not extract the values
mask = (
    (azimuth_ds['latitude'] >= lamin) & 
    (azimuth_ds['latitude'] <= lamax) & 
    (azimuth_ds['longitude'] >= lomin) & 
    (azimuth_ds['longitude'] <= lomax)
)

print(mask)
print(mask.values[:5])
subset_la = azimuth_ds['latitude'].isel(gpi=mask.values)
subset_lo = azimuth_ds['longitude'].isel(gpi=mask.values)
#print(ascat_ds)
print(np.shape(subset_la))

print('swath')
print(type(ascat_ds['swath_indicator']))
print('ncoords ', np.shape(ascat_ds['latitude']))

grid = FibGrid(12.5) # 12.5km because H121 is 12.5km spacing 
#vienna_lat, vienna_lon = 48.2, 16.4 # Vienna 
#vienna_gpi = grid.find_nearest_gpi(vienna_lon, vienna_lat)
'''



