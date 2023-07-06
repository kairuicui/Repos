################################################################################
# Module: utils.py
# Reusable functions and methods used throughout the simulator
# Rafal Kucharski @ TU Delft
################################################################################

import pandas as pd
from dotmap import DotMap
import math
import random
import numpy as np
import os
import h3
import folium


#from osmnx.distance import get_nearest_node
from osmnx.distance import nearest_nodes
from geojson.feature import Feature, FeatureCollection
import osmnx as ox
import networkx as nx
import json
from scipy.spatial.distance import euclidean


from .traveller import travellerEvent
from .driver import driverEvent


def rand_node(df):
    # returns a random node of a graph
    return df.loc[random.choice(df.index)].name


def generic_generator(generator, n):
    # to create multiple passengers/vehicles/etc
    return pd.concat([generator(i) for i in range(1, n + 1)], axis=1, keys=range(1, n + 1)).T


def empty_series(df, name=None):
    # returns empty Series from a given DataFrame, to be used for consistency of adding new rows to DataFrames
    if name is None:
        name = len(df.index) + 1
    return pd.Series(index=df.columns, name=name)


def initialize_df(df):
    # deletes rows in DataFrame and leaves the columns and index
    # returns empty DataFrame
    if type(df) == pd.core.frame.DataFrame:
        cols = df.columns
    else:
        cols = list(df.keys())
    df = pd.DataFrame(columns=cols)
    df.index.name = 'id'
    return df


def get_config(path, root_path=None, set_t0=False):
    """
    reads a .json file with MaaSSim configuration
    use as: params = get_config(config.json)
    :param path:
    :param root_path: adjust the paths with the main path while reading (used mainly for Travis tests on linux server)
    :param set_t0: adjust the t0 string to pandas datetime
    :return: params DotMap
    """
    with open(path) as json_file:
        data = json.load(json_file)
        params = DotMap(data)
    if root_path is not None:
        params.paths.G = os.path.join(root_path, params.paths.G)  # graphml of a current .city
        params.paths.skim = os.path.join(root_path, params.paths.skim)  # csv with a skim between the nodes of the .city
    if set_t0:
        params.t0 = pd.to_datetime(params.t0)
    return params


def save_config(_params, path=None):
    if path is None:
        path = os.path.join(_params.paths.params, _params.NAME + ".json")
    with open(path, "w") as write_file:
        json.dump(_params, write_file)


def set_t0(_params, now=True):
    if now:
        _params.t0 = pd.Timestamp.now().floor('1s')
    else:
        _params.t0 = pd.to_datetime(_params.t0)
    return _params


def networkstats(inData):
    """
    for a given network calculates it center of gravity (avg of node coordinates),
    gets nearest node and network radius (75th percentile of lengths from the center)
    returns a dictionary with center and radius
    """
    center_x = pd.DataFrame((inData.G.nodes(data='x')))[1].mean()
    center_y = pd.DataFrame((inData.G.nodes(data='y')))[1].mean()

    #nearest = _node(inData.G, (center_y, center_x))
    nearest = nearest_nodes(inData.G, center_x, center_y)
    ret = DotMap({'center': nearest, 'radius': inData.skim[nearest].quantile(0.75)})
    return ret


def load_G(_inData, _params=None, stats=True, set_t=True):
    # loads graph and skim from a params paths
    if set_t:
        _params = set_t0(_params)
    _inData.G = ox.load_graphml(_params.paths.G)
    _inData.nodes = pd.DataFrame.from_dict(dict(_inData.G.nodes(data=True)), orient='index')
    skim = pd.read_csv(_params.paths.skim, index_col='Unnamed: 0')
    skim.columns = [int(c) for c in skim.columns]
    _inData.skim = skim
    if stats:
        _inData.stats = networkstats(_inData)  # calculate center of network, radius and central node
    return _inData


def download_G(inData, _params, make_skims=True):
    # uses osmnx to download the graph
    print('Downloading network for {} witn osmnx'.format(_params.city))
    inData.G = ox.graph_from_place(_params.city, network_type='drive')
    inData.nodes = pd.DataFrame.from_dict(dict(inData.G.nodes(data=True)), orient='index')
    if make_skims:
        inData.skim_generator = nx.all_pairs_dijkstra_path_length(inData.G,
                                                                  weight='length')
        inData.skim_dict = dict(inData.skim_generator)  # filled dict is more usable
        inData.skim = pd.DataFrame(inData.skim_dict).fillna(_params.dist_threshold).T.astype(
            int)  # and dataframe is more intuitive

    return inData


def save_G(inData, _params, path=None):
    # saves graph and skims to files
    ox.save_graphml(inData.G, filepath=_params.paths.G)
    inData.skim.to_csv(_params.paths.skim, chunksize=20000000)



def generate_hex(inData,_params,APERTURE_SIZE=8):   #self_defined
    AMS = pd.read_csv(_params.paths.skim,nrows = 3)
    node = pd.DataFrame(AMS.columns[1:])
    node.columns = ['node']
    print(len(node['node']))
    for i in range(len(node['node'])-1,-1,-1):        #delete nodes not existing
        node['node'].iloc[i] = int(node['node'].iloc[i])
        if node['node'].iloc[i] not in inData.nodes.index:
            node = node.drop(i)
            #neighbors = list(inData.G.neighbors(node['node'].iloc[i]))  
            #lat = inData.G.nodes[node['node'].iloc[i]].y
            #lng = inData.G.nodes[node['node'].iloc[i]].x
            #neighbors = nearest_nodes(inData.G, lat = lat, lng = lng, num=10, return_sorted=True)
            #print(neighbors)
            #for k in neighbors:
            #    if k in inData.nodes.index:
            #        node['node'].iloc[i] = k
    node = node.reindex()
    print(len(node['node']))

    node['x'] = node.apply(lambda row: inData.nodes.loc[row.node].x, axis = 1)      #convert all nodes to hashes
    node['y'] = node.apply(lambda row: inData.nodes.loc[row.node].y, axis = 1)
    node['hex_o_{}'.format(APERTURE_SIZE)] = node.apply(lambda row: h3.geo_to_h3(row.y,row.x,APERTURE_SIZE),axis = 1) 

    ref = pd.concat([node['hex_o_{}'.format(APERTURE_SIZE)],node['x']],axis = 1)
    ref = pd.concat([ref,node['y']],axis = 1)
    ref = pd.concat([ref,node['node']],axis = 1)
    ref.columns = ['hash','x','y','node']  
    
    group_keys = ref.groupby('hash').groups.keys()
    centers = pd.DataFrame()
    for m in group_keys:
        df = ref[ref['hash']==m]
        vertices = h3.h3_to_geo_boundary(m)
        center = (sum(v[0] for v in vertices) / len(vertices), sum(v[1] for v in vertices) / len(vertices))
        dist = df.apply(lambda row: euclidean(center,(row['y'],row['x'])),axis = 1)
        df['dist'] = dist
        minidx = df['dist'].idxmin()
        min = df.loc[minidx]
        centers = pd.concat([centers,min],axis = 1)
    centers = centers.T
    centers.index = range(0,len(centers))


    col_geom = 'hex_o_{}'.format(APERTURE_SIZE)                 #plot base map
    hexes = pd.Series(list(set(list(node[col_geom].unique())+list(node[col_geom].unique())))).to_frame(col_geom)
    hexes = hexes.set_index(col_geom)
    hexes[col_geom] = hexes.index.copy()
    hexes['geom'] = hexes.apply(lambda x: {"type": "Polygon","coordinates": [h3.h3_to_geo_boundary(h = x[col_geom], geo_json = True)]}, axis = 1)

    list_features = []
    for i, row in hexes.iterrows(): 
        feature = Feature(geometry = row["geom"],
                          id = row[col_geom],
                          properties = {"resolution": 9})
        list_features.append(feature)
    feature_collection = FeatureCollection(list_features)
    
    geojson_hexes = json.dumps(feature_collection)
    CENTER = list(inData.nodes.loc[inData.stats.center][['y','x']].values)
    tile = 'cartodbpositron'
    base_map = folium.Map(location=CENTER, zoom_start=12,tiles = tile,zoom_control=False)
    folium.GeoJson(feature_collection, style_function=lambda x: {'fillColor': 'yellow',
                                                             'color': 'black',
                                                             'weight': 1,
                                                             'fillOpacity': 0.1,
                                                             'opacity':0.1}).add_to(base_map)
    for ind,row in centers.iterrows():    
        folium.Marker(location = [row['y'],row['x']],tooltip = row['hash']).add_to(base_map)

    #base_map.save('basemap.html')
    inData.map = base_map   
    inData.centers = centers
    inData.ref = ref
    inData.learn = {}
    return inData

def mark_on_map(ori_map,hash,color = 'red'):
    base_map = ori_map
    hexes = pd.DataFrame(hash)
    col_geom = 'hex_o_{}'.format(8)
    hexes.columns = [col_geom]
    hexes = hexes.set_index(col_geom)
    hexes[col_geom] = hexes.index.copy()
    hexes['geom'] = hexes.apply(lambda x: {"type": "Polygon","coordinates": [h3.h3_to_geo_boundary(h = x[col_geom], geo_json = True)]}, axis = 1)

    list_features = []
    for i, row in hexes.iterrows(): 
        feature = Feature(geometry = row["geom"],
                            id = row[col_geom],
                            properties = {"resolution": 9})
        list_features.append(feature)
    feature_collection = FeatureCollection(list_features)
    geojson_hexes = json.dumps(feature_collection)

    folium.GeoJson(feature_collection, style_function=lambda x: {'fillColor': color,
                                                             'color': color,
                                                             'weight': 1,
                                                             'fillOpacity': 0.1,
                                                             'opacity':0.1}).add_to(base_map) 
    #inData.map = base_map
    return base_map


def generate_vehicles(_inData, nV):
    """
    generates single vehicle (database row with structure defined in DataStructures)
    index is consecutive number if dataframe
    position is random graph node
    status is IDLE
    """
    vehs = list()
    for i in range(nV + 1):
        vehs.append(empty_series(_inData.vehicles, name=i))

    vehs = pd.concat(vehs, axis=1, keys=range(1, nV + 1)).T
    vehs.event = driverEvent.STARTS_DAY
    vehs.platform = 0
    vehs.shift_start = 0
    vehs.shift_end = 60 * 60 * 24
    vehs.pos = vehs.pos.apply(lambda x: int(rand_node(_inData.nodes)))

    return vehs

def generate_demand(_inData, _params=None, avg_speed=False):
    # generates nP requests with a given temporal and spatial distribution of origins and destinations
    # returns _inData with dataframes requests and passengers populated.

    try:
        _params.t0 = pd.to_datetime(_params.t0)
    except:
        pass
    mul = 3
    df = pd.DataFrame(index=np.arange(0, _params.nP*mul), columns=_inData.passengers.columns)
    df.status = travellerEvent.STARTS_DAY
    df.pos = _inData.nodes.sample(_params.nP*mul).index  # df.pos = df.apply(lambda x: rand_node(_inData.nodes), axis=1)
    
    requests = pd.DataFrame(index=df.index, columns=_inData.requests.columns)
    distances = _inData.skim[_inData.stats['center']].to_frame().dropna()  # compute distances from center
    distances.columns = ['distance']
    distances = distances[distances['distance'] < _params.dist_threshold]
    # apply negative exponential distributions
    distances['p_origin'] = distances['distance'].apply(lambda x:
                                                        math.exp(
                                                            _params.demand_structure.origins_dispertion * x))

    distances['p_destination'] = distances['distance'].apply(
        lambda x: math.exp(_params.demand_structure.destinations_dispertion * x))
    if _params.demand_structure.temporal_distribution == 'uniform':
        treq = np.random.uniform(-_params.simTime * 60 * 60 / 2, _params.simTime * 60 * 60 / 2,
                                 _params.nP*mul)  # apply uniform distribution on request times
    elif _params.demand_structure.temporal_distribution == 'normal':
        treq = np.random.normal(_params.simTime * 60 * 60 / 2,
                                _params.demand_structure.temporal_dispertion * _params.simTime * 60 * 60 / 2,
                                _params.nP*mul)  # apply normal distribution on request times
    else:
        treq = None
    requests.treq = [_params.t0 + pd.Timedelta(int(_), 's') for _ in treq]
    requests.origin = list(
        distances.sample(_params.nP*mul, weights='p_origin', replace=True).index)  # sample origin nodes from a distribution
    requests.destination = list(distances.sample(_params.nP*mul, weights='p_destination',
                                                 replace=True).index)  # sample destination nodes from a distribution
    
    requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    requests = requests[requests['dist']>=2000]
    df = pd.DataFrame([df.loc[i] for i in requests.index])
    requests = requests.reset_index()
    df = df.reset_index()
    requests = requests.iloc[0:_params.nP,:] 
    df = df.iloc[0:_params.nP,:] 

    _inData.passengers = df

    
    print(len(requests))
    while len(requests[requests.dist >= _params.dist_threshold]) > 0:
        requests.origin = requests.apply(lambda request: (distances.sample(1, weights='p_origin').index[0]
                                                          if request.dist >= _params.dist_threshold else
                                                          request.origin),
                                         axis=1)
        requests.destination = requests.apply(lambda request: (distances.sample(1, weights='p_destination').index[0]
                                                               if request.dist >= _params.dist_threshold else
                                                               request.destination),
                                              axis=1)
        requests.dist = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)

    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    # requests.ttrav = pd.to_timedelta(requests.ttrav)
    if avg_speed:
        requests.ttrav = (pd.to_timedelta(requests.ttrav) / _params.speeds.ride).dt.floor('1s')
    requests.tarr = [request.treq + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    #requests.index = df[df.loc[0:_params.nP,:]].index
    requests = requests.reset_index(drop = True)
    
    requests.pax_id = requests.index
    print(requests.pax_id)
    requests.shareable = False

    _inData.requests = requests
    _inData.passengers.pos = _inData.requests.origin

    _inData.passengers.platforms = _inData.passengers.platforms.apply(lambda x: [0])

    return _inData



def read_requests_csv(inData, path):
    from MaaSSim.data_structures import structures
    inData.requests = pd.read_csv(path, index_col=1)
    inData.requests.treq = pd.to_datetime(inData.requests.treq)
    inData.requests['pax_id'] = inData.requests.index.copy()
    inData.requests.ttrav = pd.to_timedelta(inData.requests.ttrav)
    inData.passengers = pd.DataFrame(index=inData.requests.index, columns=structures.passengers.columns)
    inData.passengers['pax_id'] = inData.passengers.index.copy()
    inData.passengers.pos = inData.requests.origin.copy()
    inData.passengers.platforms = inData.passengers.platforms.apply(lambda x: [0])
    return inData

def read_vehicle_positions(inData, path):
    inData.vehicles = pd.read_csv(path, index_col=0)
    return inData


def make_config_paths(params, main=None, rel = False):
    # call it whenever you change a city name, or main path
    if main is None:
        if rel:
            main = '../..'
        else:
            main = os.path.join(os.getcwd(), "../..")
    if rel:
        params.paths.main = main
    else:
        params.paths.main = os.path.abspath(main)  # main repo folder


    params.paths.data = os.path.join(params.paths.main, 'data')  # data folder (not synced with repo)
    params.paths.params = os.path.join(params.paths.data, 'configs')
    params.paths.postcodes = os.path.join(params.paths.data, 'postcodes',
                                          "PC4_Nederland_2015.shp")  # PCA4 codes shapefile
    params.paths.albatross = os.path.join(params.paths.data, 'albatross')  # albatross data
    params.paths.sblt = os.path.join(params.paths.data, 'sblt')  # sblt results
    params.paths.G = os.path.join(params.paths.data, 'graphs',
                                  params.city.split(",")[0] + ".graphml")  # graphml of a current .city
    params.paths.skim = os.path.join(params.paths.main, 'data', 'graphs', params.city.split(",")[
        0] + ".csv")  # csv with a skim between the nodes of the .city
    params.paths.NYC = os.path.join(params.paths.main, 'data',
                                    'fhv_tripdata_2018-01.csv')  # csv with a skim between the nodes of the .city
    return params


def prep_supply_and_demand(_inData, params):
    _inData = generate_demand(_inData, params, avg_speed=True)
    _inData.vehicles = generate_vehicles(_inData, params.nV)
    _inData.vehicles.platform = _inData.vehicles.apply(lambda x: 0, axis=1)
    _inData.passengers.platforms = _inData.passengers.apply(lambda x: [0], axis=1)
    _inData.requests['platform'] = _inData.requests.apply(lambda row: _inData.passengers.loc[row.name].platforms[0],
                                                          axis=1)

    _inData.platforms = initialize_df(_inData.platforms)
    _inData.platforms.loc[0] = [1, 'Platform', 1]
    return _inData


#################
# PARALLEL RUNS #
#################


def test_space():
    # to see if code works
    test_space = DotMap()
    test_space.nP = [30, 40]  # number of requests per sim time
    test_space.nV = [10, 20]  # number of requests per sim time
    return test_space


def slice_space(s, replications=1, _print=False):
    # util to feed the np.optimize.brute with a search space
    def sliceme(l):
        return slice(0, len(l), 1)

    ret = list()
    sizes = list()
    size = 1
    for key in s.keys():
        ret += [sliceme(s[key])]
        sizes += [len(s[key])]
        size *= sizes[-1]
    if replications > 1:
        sizes += [replications]
        size *= sizes[-1]
        ret += [slice(0, replications, 1)]
    if _print:
        print('Search space to explore of dimensions {} and total size of {}'.format(sizes, size))
    return tuple(ret)


def collect_results(path):
    from pathlib import Path
    import zipfile
    collections = DotMap()
    first = True
    for archive in Path(path).rglob('*.zip'):
        zf = zipfile.ZipFile(archive)
        if first:
            for file in zf.namelist():
                collections[file[:-4]] = list()
            first = False
        for file in zf.namelist():
            df = pd.read_csv(zf.open(file))
            for key in archive.stem.split('-')[1:]:
                field, value = key.split('_')
                df[field] = value

            collections[file[:-4]].append(df)
    for key in collections.keys():
        collections[key] = pd.concat(collections[key])
    return collections
