
import pandas as pd
import geopandas as gpd
import shapely

def bd_preprocess(building):
    def bd_preprocess_old(buildings, height=''):
        '''
        Preprocess building data, so that we can perform shadow calculation.
        Remove empty polygons and convert multipolygons into polygons.

        Parameters
        --------------
        buildings : GeoDataFrame
            Buildings.
        height : string
            Column name of building height(meter).

        Return
        ----------
        allbds : GeoDataFrame
            Polygon buildings
        '''
        
        buildings['geometry'] = buildings.buffer(0)
        buildings = buildings[buildings.is_valid].copy()
        if height!='':
            # 建筑高度筛选
            buildings[height] = pd.to_numeric(buildings[height], errors='coerce')
            buildings = buildings[buildings[height]>0].copy()

        polygon_buildings = buildings[buildings['geometry'].apply(
            lambda r:type(r) == shapely.geometry.polygon.Polygon)]
        multipolygon_buildings = buildings[buildings['geometry'].apply(
            lambda r:type(r) == shapely.geometry.multipolygon.MultiPolygon)]
        allbds = []
        for j in range(len(multipolygon_buildings)):
            r = multipolygon_buildings.iloc[j]
            singlebd = gpd.GeoDataFrame()
            singlebd['geometry'] = list(r['geometry'].geoms)
            for i in r.index:
                if i != 'geometry':
                    singlebd[i] = r[i]
            allbds.append(singlebd)
        allbds.append(polygon_buildings)
        allbds = pd.concat(allbds)
        if len(allbds) > 0:
            allbds = gpd.GeoDataFrame(allbds)
            allbds['building_id'] = range(len(allbds))
            allbds['geometry'] = allbds.buffer(0)
        else:
            allbds = gpd.GeoDataFrame()
        allbds.crs = {'init': 'epsg:4326'}
        return allbds
    
    building = building.groupby(['height']).apply(lambda r:r.unary_union).reset_index()
    building.columns = ['height','geometry']
    building = gpd.GeoDataFrame(building,geometry = 'geometry')
    building = bd_preprocess_old(building)
    return building