################################################################################
# Module: shared.py
# Integration with ExMAS
# Exact Matching of Attractive Shared rides (ExMAS) for system-wide strategic evaluations
# Rafal Kucharski @ TU Delft
################################################################################

import pandas as pd
import random
from ExMAS.main import matching
import ExMAS


def prep_shared_rides(_inData, sp, _print=False):
    """
    Determines which requests are shareable,
    computes the matching with ExMAS,
    generates schedule of shared (or non shared) rides
    :param _inData:
    :param sp:
    :param sblt: external ExMas library (import ExMAS.main as sblt) optional
    :param _print:
    :return:
    """
    #sp = _params.shareability
    def set_indexes(row):
        # determines which rides contain this request
        return _inData.sblts.rides[_inData.sblts.rides.apply(lambda x: row.pax_id in x.indexes, axis=1)].index.to_list()

    sp.share = sp.get('share', 0)  # share of shareable rides
    requests = _inData.requests

    if sp.shape == 0:
        requests.shareable = False  # all requests are not shareable
    elif sp.share == 1:
        requests.shareable = True  # all requests are shareable
    else:  # mixed - not fully tested, can be unstable
        requests.shareable = requests.apply(lambda x:
                                            False if random.random() >= sp.share
                                            else True, axis=1)

    if requests[requests.shareable].shape[0] == 0:  # no sharing
        _inData.requests['ride_id'] = _inData.requests.index.copy()  # rides are schedules, we do not share
        _inData.requests['position'] = 0  # everyone has first position
        # _inData.requests['sim_schedule'] = _inData.requests.apply(lambda x: , axis=1)

    else:  # sharing
        if sp.without_matching:  # was the shareability graph comouted before?
            _inData = matching(_inData, sp, plot=False)  # if so, do only matching
        else:
            _inData = ExMAS.main(_inData, sp, plot=False)  # compute graph and do matching

        # prepare schedules
        schedule = _inData.sblts.schedule
        r = _inData.sblts.requests
        schedule['nodes'] = schedule.apply(lambda s: [None] + list(r.loc[s.indexes_orig].origin.values) +
                                                     list(r.loc[s.indexes_dest].destination.values), axis=1)

        schedule['req_id'] = schedule.apply(lambda s: [None] + s.indexes_orig + s.indexes_dest, axis=1)

        _inData.requests['ride_id'] = r.ride_id  # store ride index in requests for simulation
        _inData.requests['position'] = r.position  # store ride index in requests for simulation
        _inData.sblts.schedule['sim_schedule'] = _inData.sblts.schedule.apply(lambda x: make_schedule_shared(x), axis=1)

        # rides preparation for pricing Usman
        # prepare rides
        rides = _inData.sblts.rides
        rides['degree'] = rides.apply(lambda row: len(row.indexes), axis=1)
        r = _inData.sblts.requests
        rides['nodes'] = rides.apply(lambda s: [None] + list(r.loc[s.indexes_orig].origin.values) +
                                                     list(r.loc[s.indexes_dest].destination.values), axis=1)

        rides['req_id'] = rides.apply(lambda s: [None] + s.indexes_orig + s.indexes_dest, axis=1)

        _inData.requests['rides'] = _inData.requests.apply(set_indexes, axis=1) # assign rides to request (list of rides containing this request)
        _inData.requests['position'] = 0  # this way all travllers will be triggered
        
        _inData.sblts.rides['sim_schedule'] = _inData.sblts.rides.apply(lambda x: make_schedule_shared(x), axis=1) # do the schedules for all the rides

        _inData.sblts.rides['ttrav'] = _inData.sblts.rides.apply(lambda row: sum(row.times[1:]),
                                                                 axis=1)
        _inData.sblts.rides['dist'] = _inData.sblts.rides.apply(lambda row:
                                                                row.ttrav*sp.avg_speed/1000, axis=1) # in km
        _inData.sblts.rides['fare'] = _inData.sblts.rides.apply(lambda row: sum(_inData.requests.loc[p].dist/1000 for p in row.indexes)*sp.price*(1-sp.shared_discount) if row.degree>1
                                      else sum(_inData.requests.loc[p].dist/1000 for p in row.indexes)*sp.price*1, axis=1) # in Euro# in Euro
        _inData.sblts.rides['commission'] = _inData.sblts.rides.apply(lambda row: row.fare*sp.comm_rate, axis=1)
        _inData.sblts.rides['driver_revenue'] = _inData.sblts.rides['fare'] - _inData.sblts.rides['commission']
        
    def set_sim_schedule(x):
        if not x.shareable:
            return make_schedule_nonshared([x])
        elif 'platform' in x and x.platform == -1:
            return make_schedule_nonshared([x])
        else:
            return _inData.sblts.schedule.loc[x.ride_id].sim_schedule

    _inData.requests['sim_schedule'] = _inData.requests.apply(lambda x: set_sim_schedule(x), axis=1)


    
    

    _inData.schedules_queue = pd.DataFrame([[i, _inData.schedules[i].node[1]]
                                            for i in _inData.schedules.keys()],
                                           columns=[0, 'origin']).set_index(0)

    return _inData


def make_schedule_nonshared(requests):
    """
    preares the schedule for a non shared rides
    :param requests: inData.requests
    :return: schedule
    """
    columns = ['node', 'time', 'req_id', 'od']
    degree = 2 * len(requests) + 1
    df = pd.DataFrame(None, index=range(degree), columns=columns)
    nodes = [None] + [r.origin for r in requests] + [r.destination for r in requests]
    df.node = nodes
    df.req_id = [None] + [r.name for r in requests] * 2

    df.od = [None] + ['o'] * len(requests) + ['d'] * len(requests)

    return df


def make_schedule_shared(row):
    """
    prpepares a schedule of a shared ride for simulation
    schedule is a sequence of visited nodes in time
    :param row: single shared ride from ExMAS
    :return:
    """
    columns = ['node', 'time', 'req_id', 'od']
    degree = int(2 * row.degree + 1)
    schedule = pd.DataFrame(None, index=range(degree), columns=columns)

    nodes = row.nodes
    schedule.req_id = row.req_id
    schedule.node = nodes

    schedule.od = [None] + ['o'] * row.degree + ['d'] * row.degree

    return schedule
