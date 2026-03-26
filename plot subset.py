import xarray as xr 
import numpy as np 
from fibgrid.realization import FibGrid
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys

def make_map(bbox, projection=ccrs.PlateCarree()):
    font = {  'size': 16  }
    fig, ax = plt.subplots(figsize=(12, 10),
                           subplot_kw=dict(projection=projection))
    ax.set_extent(bbox)
    #ax.add_feature(LAND, facecolor='0.75')
    ax.coastlines(resolution='50m')
    gl = ax.gridlines(draw_labels=True)
    #gl = ax.gridlines()
    gl.top_labels = False
    gl.right_labels = False
    #gl.xlocator = mticker.FixedLocator([134, 135, 140, 145, 150])
    #gl.ylocator = mticker.FixedLocator([10, 15, 20, 25, 30, 35])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}
    return fig, ax

rivers = cfeature.NaturalEarthFeature(
    category="physical",
    name="rivers_lake_centerlines",
    scale="10m",
    facecolor="none",      # we only want lines
    edgecolor="blue"       # color of rivers
)
lakes = cfeature.NaturalEarthFeature(
    category="physical",
    name="lakes",
    scale="10m",
    facecolor="none",      # or use a color if you want filled lakes
    edgecolor="blue"
)

fill_value = -2.1474836e+09
#import cartopy.crs as ccrs
#import matplotlib.pyplot as plt

# Vienna
lomax = 17
lomin = 15.5
lamax = 48.5
lamin = 47.5
'''
# Lisbon
lomax = -5.5
lomin = -11.5
lamax = 42.5
lamin = 35.5
'''
ascat_ds = xr.open_zarr("/home/mpanfilo/radar01/RADAR/ascat/ascat_resample/ascat_METOPABC_12.5_ts_v2.zarr")
xr.set_options(display_max_rows=1000, display_expand_data=True)
print(ascat_ds)
azimuth_ds = xr.open_zarr("/home/mpanfilo/radar01/RADAR/ascat/metop_abc/azimuth.zarr")
print(azimuth_ds)

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

subset_fitcoeff = azimuth_ds['fitcoeff'].isel(gpi=mask.values)
print(np.shape(subset_fitcoeff))
print(np.shape(subset_la))
#sys.exit(0)


# Coefficients define dby Roland: sigma_0 = b0 + b1*inc_angle + b2
b0 = subset_fitcoeff.values[:,1,2] 
b1 = subset_fitcoeff.values[:,1,1]
b2 = subset_fitcoeff.values[:,1,0]

for i in range(len(b0)):
    x = np.linspace(30, 60)
    xr = 40
    y = b0[i] + (x - xr) * b1[i] + (x - xr)**2 * b2[i]
    plt.plot(x, y)

plt.savefig('regression_examples.jpg')
plt.close()
sys.exit(0)

inc_min = 35
inc_max = 65
der_th_min = b1 + 2 * inc_min * b2 # first derivative at minimum incidence angle
der_th_max = b1 + 2 * inc_max * b2 # first derivative at maximum incidence angle

mask_der = np.logical_and(der_th_min*der_th_max<0,
                        der_th_min>0,
                        b0 != fill_value)
mask_fill = b2 != fill_value
mask = mask_fill

#sys.exit(0)
font = {  'size': 16  }

values = [b0, b1, b2, der_th_min]
value_names = ['b0', 'b1', 'b2', 'derivative_th_min']

for value, value_name in zip(values, value_names):
    fig, ax = make_map([lomin, lomax, lamin, lamax], projection=ccrs.PlateCarree())
    cs31 = ax.scatter(subset_lo[mask], subset_la[mask], s = 50, c = value[mask], cmap = 'jet')
    cbar = plt.colorbar(cs31, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    plt.title('configuration 1, ' + value_name)

    ax.add_feature(rivers)
    ax.add_feature(lakes)

    plt.savefig('figures/test_plots_coeffs_bi_set1/subset_lisbon_' + value_name + '.jpg')
    plt.close()