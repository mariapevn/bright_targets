import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


class MapPlotter:
    def __init__(self, bbox, projection=ccrs.PlateCarree()):
        self.bbox = bbox
        self.projection = projection

    def create_map(self):
        fig, ax = plt.subplots(figsize=(12, 10),
                               subplot_kw=dict(projection=self.projection))
        ax.set_extent(self.bbox)
        ax.coastlines(resolution='50m')

        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

        return fig, ax

    def add_features(self, ax):
        rivers = cfeature.NaturalEarthFeature(
            category="physical",
            name="rivers_lake_centerlines",
            scale="10m",
            facecolor="none",
            edgecolor="blue"
        )
        lakes = cfeature.NaturalEarthFeature(
            category="physical",
            name="lakes",
            scale="10m",
            facecolor="none",
            edgecolor="blue"
        )

        ax.add_feature(rivers)
        ax.add_feature(lakes)


class AzimuthProcessor:
    def __init__(self, path, bbox, fill_value):
        self.ds = xr.open_zarr(path)
        self.lomin, self.lomax, self.lamin, self.lamax = bbox
        self.fill_value = fill_value

    def subset(self):
        mask = (
            (self.ds['latitude'] >= self.lamin) &
            (self.ds['latitude'] <= self.lamax) &
            (self.ds['longitude'] >= self.lomin) &
            (self.ds['longitude'] <= self.lomax)
        )

        self.lat = self.ds['latitude'].isel(gpi=mask.values)
        self.lon = self.ds['longitude'].isel(gpi=mask.values)
        self.fitcoeff = self.ds['fitcoeff'].isel(gpi=mask.values)

    def get_coefficients(self, config):
        b0 = self.fitcoeff.values[:, config, 2]
        b1 = self.fitcoeff.values[:, config, 1]
        b2 = self.fitcoeff.values[:, config, 0]

        mask = b2 != self.fill_value
        return b0[mask], b1[mask], b2[mask], self.lat[mask], self.lon[mask]

    def get_mask_bright_targets(self, config):
        b0, b1, b2, lat, lon = processor.get_coefficients(config)

        inc_min = 30
        inc_max = 65
        der_th_min = b1 + 2 * (inc_min-40) * b2 # first derivative at minimum incidence angle
        der_th_max = b1 + 2 * (inc_max-40) * b2 # first derivative at maximum incidence angle
        mask_der = np.logical_and(der_th_max<0,
                                der_th_min>0)
        return(mask_der)

class RegressionPlotter:
    def __init__(self):
        pass

    def plot_all_configs(self, fitcoeff, filename):
        fig, axes = plt.subplots(4, 4, figsize=(14, 12))
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.5, wspace=0.4)

        for config in range(13):
            b0 = fitcoeff[:, config, 2]
            b1 = fitcoeff[:, config, 1]
            b2 = fitcoeff[:, config, 0]

            mask = b2 != fill_value
            b0, b1, b2 = b0[mask], b1[mask], b2[mask]

            for i in range(len(b0)):
                x = np.linspace(30, 60)
                xr = 40
                y = b0[i] + (x - xr) * b1[i] + (x - xr) ** 2 * b2[i]
                axes[config].plot(x, y)

            axes[config].set_title(f"Config {config}")

        for j in range(13, 16):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_map(self, lon, lat, values, title, bbox, filename, mask=None):
        map_plotter = MapPlotter(bbox)
        fig, ax = map_plotter.create_map()

        if mask is not None:
            lon = lon[mask]
            lat = lat[mask]
            values = values[mask]

        sc = ax.scatter(lon, lat, c=values, cmap='jet', s=50)
        plt.colorbar(sc, ax=ax)
        plt.title(title)

        map_plotter.add_features(ax)

        plt.savefig(filename)
        plt.close()    


# =========================
# MAIN SCRIPT
# =========================

if __name__ == "__main__":

    fill_value = -2.1474836e+09

    # Belgium bbox
    bbox = (2.3, 6.6, 49.4, 51.5)

    processor = AzimuthProcessor(
        path="/home/mpanfilo/radar01/RADAR/ascat/metop_abc/azimuth.zarr",
        bbox=bbox,
        fill_value=fill_value
    )

    processor.subset()

    plotter = RegressionPlotter()

    # Plot all configurations
    plotter.plot_all_configs(
        processor.fitcoeff.values,
        "regression_all_configs.jpg"
    )

    # Example: configuration 1
    b0, b1, b2, lat, lon = processor.get_coefficients(config=1)

    inc_min, inc_max = 30, 65
    der_min = b1 + 2 * (inc_min - 40) * b2
    der_max = b1 + 2 * (inc_max - 40) * b2

    values = {
        "b0": b0,
        "b1": b1,
        "b2": b2,
        "der_min": der_min,
        "der_max": der_max
    }

    for name, val in values.items():
        plotter.plot_map(
            lon, lat, val,            
            f"Config 1 - {name}",
            bbox,
            f"{name}.jpg",
            mask=None
        )

    
    mask = processor.get_mask_bright_targets(3)
    plotter.plot_map(
            lon, lat, b2,            
            f"Mask targets 1 - b2",
            bbox,
            "mask_targets_b2_config.jpg",
            mask
        )