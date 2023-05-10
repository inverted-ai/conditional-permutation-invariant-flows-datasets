import numpy as np

recenter_offsets = {
    "DR_CHN_Merging_ZS"      : np.array([1070.7115, 955.2069]),
    "DR_CHN_Roundabout_LN"   : np.array([991.3134, 1002.7626]),
    "DR_DEU_Merging_MT"      : np.array([944.3038, 1006.1682]),
    "DR_DEU_Roundabout_OF"   : np.array([999.4451, 989.8357]),
    "DR_USA_Intersection_EP0": np.array([1003.7960, 994.3797]),
    "DR_USA_Intersection_EP1": np.array([1029.4236, 991.0101]),
    "DR_USA_Intersection_GL" : np.array([978.9284, 985.4507]),
    "DR_USA_Intersection_MA" : np.array([1026.6316, 1003.1119]),
    "DR_USA_Roundabout_EP"   : np.array([1019.2935, 1012.0157]),
    "DR_USA_Roundabout_FT"   : np.array([1015.1415, 999.9953]),
    "DR_USA_Roundabout_SR"   : np.array([993.7158, 1021.8036]),
}

# NOTE: the interaction track contains some tracks that are rotated, this is used to correct them
rotated_tracks = {
    ('DR_CHN_Merging_ZS', 5)      : [474],
    ('DR_CHN_Merging_ZS', 8)      : [127, 406, 526, 820, 878],
    ('DR_CHN_Merging_ZS', 10)     : [86, 578, 601],
    ('DR_DEU_Merging_MT', 14)     : [9],
    ('DR_DEU_Roundabout_OF', 6)   : [71],
    ('DR_DEU_Roundabout_OF', 9)   : [40, 65],
    ('DR_DEU_Roundabout_OF', 10)  : [8],
    ('DR_USA_Intersection_EP1', 1): [39],
    ('DR_USA_Intersection_GL', 4) : [72],
    ('DR_USA_Intersection_GL', 8) : [12],
    ('DR_USA_Intersection_GL', 34): [196],
    ('DR_USA_Intersection_GL', 40): [37],
    ('DR_USA_Intersection_GL', 44): [43],
    ('DR_USA_Intersection_GL', 45): [25],
    ('DR_USA_Intersection_GL', 52): [81],
    ('DR_USA_Intersection_GL', 54): [58],
    ('DR_USA_Intersection_MA', 12): [86],
    ('DR_USA_Roundabout_EP', 3)   : [80],
    ('DR_USA_Roundabout_FT', 32)  : [83],
    ('DR_USA_Roundabout_FT', 40)  : [10],
    ('DR_USA_Roundabout_SR', 1)   : [118],
    ('DR_USA_Roundabout_SR', 2)   : [113],
    ('DR_USA_Roundabout_SR', 3)   : [12],
    ('DR_USA_Roundabout_SR', 5)   : [146],
    ('DR_USA_Roundabout_SR', 8)   : [36, 43, 78]
}

location_names = [
    "DR_CHN_Merging_ZS"      ,
    "DR_CHN_Roundabout_LN"   ,
    "DR_DEU_Merging_MT"      ,
    "DR_DEU_Roundabout_OF"   ,
    "DR_USA_Intersection_EP0",
    "DR_USA_Intersection_EP1",
    "DR_USA_Intersection_GL" ,
    "DR_USA_Intersection_MA" ,
    "DR_USA_Roundabout_EP"   ,
    "DR_USA_Roundabout_FT"   ,
    "DR_USA_Roundabout_SR"   ,
]
