
import pandas as pd


def get_unique_sessions(data:pd.DataFrame, session_title:str='session', prefix:str='treadmill'):
    return data[session_title].dropna()[data[session_title].dropna().str.startswith(prefix)].unique()

def get_signal(data:pd.DataFrame, session_name:str, signal_name:str,session_title:str='session'):
    return data[data[session_title] == session_name][signal_name].dropna()

def get_right_gc(data, gc_session_name):
    return get_signal(data, session_name=gc_session_name, signal_name='gcRight_HeelStrike',session_title='gcRight_session')

if __name__ == '__main__':
    imu_sample = '/media/champagne/lower_limb_dataset/v2/AB24_12_02_2018_IMU.pkl'
    grf_sample = imu_sample.replace('IMU','GRF')
    imu = pd.read_pickle(imu_sample)
    grf = pd.read_pickle(grf_sample)

    session = 'treadmill_02_01'
    gc = get_right_gc(imu, session)
    print(gc.shape)

    for sig in ['shank_Accel_X','trunk_Accel_X','shank_Gyro_X','trunk_Gyro_X']:
        print(get_signal(imu, session, sig).shape)
        print(get_signal(imu, session, sig).index.shape)