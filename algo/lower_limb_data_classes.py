import os
from itertools import product
from typing import List, Optional
from dataclasses import dataclass, field
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import glob

@dataclass
class LowerLimbData:
    """Data class for lower limb data."""
    date: str
    subject_id: str
    age: int
    gender: str
    height: float
    weight: float
    locomotion_modes: List[str]
    signal_sources: List[str]
    data: pd.DataFrame = field(repr=False)
    imu_signal_types: Optional[List[str]] = None
    grf_signal_types: Optional[List[str]] = None

    def __post_init__(self):
        self.imu_signal_types = [
            'foot.Accel', 'foot.Gyro', 'shank.Accel', 'shank.Gyro', 
            'thigh.Accel', 'thigh.Gyro', 'trunk.Accel', 'trunk.Gyro'
        ]
        self.grf_signal_types = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    @classmethod
    def from_csv(cls, signaldata_path: str, subjectInfo_path:str, locomotion_modes: Optional[List[str]] = None, signal_sources: Optional[List[str]] = None) -> 'LowerLimbData':
        if locomotion_modes is None:
            locomotion_modes = ['levelground', 'ramp', 'stair', 'treadmill']
        if signal_sources is None:
            signal_sources = ['imu', 'fp']
        
        # Parse the file path to get the subject ID and the date.
        path_parts = signaldata_path.split('/')
        subject_id = path_parts[-2]
        date = path_parts[-1].split('.')[0]  # remove file extension

        subjectInfo = pd.read_csv(signaldata_path)
        # Find the subject by ID and fill information such as age, gender, height, weight
        subject_info = pd.read_csv(subjectInfo_path)
        subject_row = subject_info[subject_info['Subject'] == subject_id].iloc[0]

        age = subject_row['Age']
        gender = subject_row['Gender']
        height = subject_row['Height']
        weight = subject_row['Weight']

        return cls(
            date=date,
            subject_id=subject_id,
            locomotion_modes=locomotion_modes,
            signal_sources=signal_sources,
            data=subjectInfo,
            age=age,
            gender=gender,
            height=height,
            weight=weight
        )

def combine_data(parent_path: str, locomotion_modes: Optional[List[str]] = None, signal_sources: Optional[List[str]] = None) -> LowerLimbData:
    if locomotion_modes is None:
        locomotion_modes = ['levelground', 'ramp', 'stair', 'treadmill']
    if signal_sources is None:
        signal_sources = ['imu', 'gcLeft', 'gcRight', 'fp']
    
    combined_person_data = pd.DataFrame()

    def read_and_process_file(file_path, session):
        data = pd.read_csv(file_path)
        data.set_index('Header', inplace=True)
        data['session'] = session
        return data

    def process_signal_source(locomotion_mode, signal_source):
        dir_path = os.path.join(parent_path, locomotion_mode, signal_source)
        signal_data = pd.DataFrame()
        files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

        for file in files:  # join per session
            file_path = os.path.join(dir_path, file)
            print("processing", file_path)
            result = read_and_process_file(file_path, file.removesuffix('.csv'))
            if signal_source == 'fp':
                result = result[::5]
            signal_data = pd.concat([signal_data, result], axis=0)
        
        return locomotion_mode, signal_source, signal_data

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_signal_source, lm, ss) for lm, ss in product(locomotion_modes, signal_sources)]
        results = [future.result() for future in futures]

    for locomotion_mode, signal_source, signal_data in results:
        persignal_data = pd.DataFrame()
        persignal_data = pd.merge(persignal_data, signal_data.drop(columns=['session']), left_index=True, right_index=True, how='outer') if not persignal_data.empty else signal_data
        persignal_data['locomotion_mode'] = locomotion_mode
        combined_person_data = pd.concat([combined_person_data, persignal_data], axis=0)
        print("current signal data size", signal_data.shape, "motion data size", persignal_data.shape)

    return combined_person_data

if __name__ == "__main__":
    base_path = "./dataset/raw/"
    subject_paths = glob.glob(os.path.join(base_path, "AB*/*_*_*"))

    for file_path in subject_paths:
        subject_id = file_path.split('/')[-2]
        date = file_path.split('/')[-1]

        data = combine_data(file_path, signal_sources=['imu', 'gcLeft', 'gcRight'])
        data.to_pickle(f"{subject_id}_{date}_IMU.pkl")
        
        data = combine_data(file_path, signal_sources=['fp'])
        data.to_pickle(f"{subject_id}_{date}_GRF.pkl")
