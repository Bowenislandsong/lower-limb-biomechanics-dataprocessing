import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import find_peaks
import itertools
import numpy as np

def get_data_for_session(imu_data, grf_data, session, locomotion='treadmill', include_imu_columns=['shank_', 'trunk_','_HeelStrike'], include_grf_columns=['Treadmill_L_v','Treadmill_R_v']):
    # print sesssions, if the session is not in the data.
    if session not in imu_data['session'].unique():
        print(f"Session {session} not found in the IMU data. Available sessions: {imu_data['session'].unique()}")
    if session not in grf_data['session'].unique():
        print(f"Session {session} not found in the GRF data. Available sessions: {grf_data['session'].unique()}")

    imu_data_filtered = imu_data[
        (imu_data['session'] == session) | 
        ('gcLeft_session' in imu_data.columns and imu_data['gcLeft_session'] == session) | 
        ('gcRight_session' in imu_data.columns and imu_data['gcRight_session'] == session)
    ]
    imu_data_filtered = imu_data_filtered[imu_data_filtered['locomotion_mode'] == locomotion]
    grf_data_filtered = grf_data[(grf_data['session'] == session)]
    grf_data_filtered = grf_data_filtered[grf_data_filtered['locomotion_mode'] == locomotion]

    # combine the data
    data = pd.merge(imu_data_filtered, grf_data_filtered, left_index=True, right_index=True, suffixes=('_imu', '_grf'))
    # Filter the columns to include only the specified columns
    imu_columns_to_include = [col for col in data.columns if any([pattern in col for pattern in include_imu_columns])]
    grf_columns_to_include = [col for col in data.columns if any([pattern in col for pattern in include_grf_columns])]
    data = data[imu_columns_to_include + grf_columns_to_include]

    return data


def segment_data(data, n_segments, n_elements):
    segment_size = len(data) // n_segments
    segmented_data = []
    for i in range(n_segments):
        segment = data[i*segment_size:(i+1)*segment_size]
        middle_index = len(segment) // 2
        start_index = max(0, middle_index - n_elements // 2)
        end_index = start_index + n_elements
        segmented_data.extend(segment[start_index:end_index])
    return segmented_data

def step_segmentation(data, window_size=100,
                      num_speeds=4, steps_per_speed=15,
                      indicator_l='gcLeft_HeelStrike', indicator_r='gcRight_HeelStrike', 
                      left_column='Treadmill_L_v', right_column='Treadmill_R_v'):
    segmented_data = []
    
    # Find heel strike events
    heel_strikes_l,_ = find_peaks(data[indicator_l])
    heel_strikes_r,_ = find_peaks(data[indicator_r])

    heel_strikes_l = segment_data(heel_strikes_l, num_speeds, steps_per_speed)
    heel_strikes_r = segment_data(heel_strikes_r, num_speeds, steps_per_speed)

    # Segment data based on heel strikes
    for l,r in zip(heel_strikes_l,heel_strikes_r):
        left,right = [],[]
        for axis in itertools.product([left_column,right_column],['x', 'y', 'z']):
            col = ''.join(axis)
            segment_l = data[col].iloc[l:l + window_size].tolist()
            segment_r = data[col].iloc[r:r + window_size].tolist()
            left.extend(segment_l)
            right.extend(segment_r)
        segmented_data.append(left + right)
    
    return np.asarray(segmented_data)