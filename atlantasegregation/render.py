"""Display data in a map of Atlanta."""
import os

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from data import get_atlanta_boundary, RACES


def save_image(gdf: gpd.GeoDataFrame, filepath: str | os.PathLike) -> None:
    """
    Save an image representing a GeoDataFrame of Atlanta to the given filepath.
    :param gdf: The GeoDataFrame of data in Atlanta, GA.
    :param filepath: The path to which the image will be saved.
    """
    atlanta_boundary = get_atlanta_boundary()

    majority_race = gdf.take([gdf.columns.get_loc(race) for race in RACES], axis=1) \
        .idxmax(axis=1, numeric_only=True)  # We only care about the majority race in this case

    plt.figure(figsize=(12, 12))
    plt.gca().set_aspect('equal')

    # Find the bounds of Atlanta
    minimum_x = np.inf
    minimum_y = np.inf
    maximum_x = -np.inf
    maximum_y = -np.inf
    for geom in atlanta_boundary.iloc[0]['geometry'].geoms:
        xs, ys = geom.exterior.xy
        plt.plot(xs, ys, 'k-')
        minx, miny, maxx, maxy = geom.bounds
        minimum_x, maximum_x = min(minimum_x, minx), max(maximum_x, maxx)
        minimum_y, maximum_y = min(minimum_y, miny), max(maximum_y, maxy)

    # Display each block with a color according to race
    for race, color in zip(RACES, mcolors.BASE_COLORS):
        points = gdf.loc[majority_race == race].to_crs(crs=atlanta_boundary.crs)['geometry']
        plt.scatter(points.x, points.y, color=color, label=race)

    # Display a legend
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', bbox_transform=plt.gcf().transFigure)
    # +/- 100 is so the border is fully displayed
    plt.xlim(minimum_x - 100, maximum_x + 100)
    plt.ylim(minimum_y - 100, maximum_y + 100)
    plt.grid(False)
    plt.axis('off')
    plt.title('Race in Atlanta, Georgia')
    plt.savefig(filepath)
