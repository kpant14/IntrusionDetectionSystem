import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
'''65 features'''
# column_name = ['load_cpuload_0',
#       'ram_usage_cpuload_0',
#       'visual_odometry_timestamp_rel_ekf2_timestamps_0',
#       'ev_hpos[0]_estimator_innovation_test_ratios_0',
#       'ev_hpos[1]_estimator_innovation_test_ratios_0',
#       'ev_vpos_estimator_innovation_test_ratios_0',
#       'ev_hvel[0]_estimator_innovation_test_ratios_0',
#       'ev_hvel[1]_estimator_innovation_test_ratios_0',
#       'ev_vvel_estimator_innovation_test_ratios_0',
#       'heading_estimator_innovation_test_ratios_0',
#       'ev_hpos[0]_estimator_innovation_variances_0',
#       'ev_hpos[1]_estimator_innovation_variances_0',
#       'ev_vpos_estimator_innovation_variances_0',
#       'ev_hvel[0]_estimator_innovation_variances_0',
#       'ev_hvel[1]_estimator_innovation_variances_0',
#       'ev_vvel_estimator_innovation_variances_0',
#       'heading_estimator_innovation_variances_0',
#       'ev_hpos[0]_estimator_innovations_0',
#       'ev_hpos[1]_estimator_innovations_0',
#       'ev_vpos_estimator_innovations_0',
#       'ev_hvel[0]_estimator_innovations_0',
#       'ev_hvel[1]_estimator_innovations_0',
#       'ev_vvel_estimator_innovations_0',
#       'heading_estimator_innovations_0',
#       'gyro_bias[0]_estimator_sensor_bias_0',
#       'gyro_bias[1]_estimator_sensor_bias_0',
#       'gyro_bias[2]_estimator_sensor_bias_0',
#       'accel_bias[0]_estimator_sensor_bias_0',
#       'accel_bias[1]_estimator_sensor_bias_0',
#       'accel_bias[2]_estimator_sensor_bias_0',
#       'covariances[0]_estimator_status_0',
#       'covariances[1]_estimator_status_0',
#       'covariances[2]_estimator_status_0',
#       'covariances[3]_estimator_status_0',
#       'covariances[4]_estimator_status_0',
#       'covariances[5]_estimator_status_0',
#       'covariances[6]_estimator_status_0',
#       'covariances[7]_estimator_status_0',
#       'covariances[8]_estimator_status_0',
#       'covariances[9]_estimator_status_0',
#       'covariances[10]_estimator_status_0',
#       'covariances[11]_estimator_status_0',
#       'covariances[12]_estimator_status_0',
#       'covariances[13]_estimator_status_0',
#       'covariances[14]_estimator_status_0',
#       'covariances[15]_estimator_status_0',
#       'pos_test_ratio_estimator_status_0',
#       'pos_horiz_accuracy_estimator_status_0',
#       'pos_vert_accuracy_estimator_status_0',
#       'mag_test_ratio_estimator_status_0',
#       'vibe[0]_estimator_status_0',
#       'vibe[1]_estimator_status_0',
#       'vibe[2]_estimator_status_0',
#       'rollspeed_integ_rate_ctrl_status_0',
#       'pitchspeed_integ_rate_ctrl_status_0',
#       'yawspeed_integ_rate_ctrl_status_0',
#       'rate_rx_telemetry_status_0',
#       'rate_tx_telemetry_status_0',
#       'accel_vibration_metric_vehicle_imu_status_0',
#       'gyro_vibration_metric_vehicle_imu_status_0',
#       'gyro_coning_vibration_vehicle_imu_status_0',
#       'eph_vehicle_local_position_0',
#       'epv_vehicle_local_position_0',
#       'evh_vehicle_local_position_0',
#       'evv_vehicle_local_position_0',]

'''32 features'''
column_name = ['load_cpuload_0',
'covariances[2]_estimator_status_0',
'covariances[4]_estimator_status_0',
'covariances[5]_estimator_status_0',
'evh_vehicle_local_position_0',
'covariances[7]_estimator_status_0',
'covariances[8]_estimator_status_0',
'covariances[9]_estimator_status_0',
'covariances[10]_estimator_status_0',
'covariances[11]_estimator_status_0',
'pos_horiz_accuracy_estimator_status_0',
'vibe[0]_estimator_status_0',
'vibe[1]_estimator_status_0',
'vibe[2]_estimator_status_0',
'yawspeed_integ_rate_ctrl_status_0',
'rate_rx_telemetry_status_0',
'accel_vibration_metric_vehicle_imu_status_0',
'gyro_vibration_metric_vehicle_imu_status_0',
'gyro_coning_vibration_vehicle_imu_status_0',
'eph_vehicle_local_position_0',
'covariances[1]_estimator_status_0',
'covariances[0]_estimator_status_0',
'covariances[6]_estimator_status_0',
'evv_vehicle_local_position_0',
'accel_bias[0]_estimator_sensor_bias_0',
'ram_usage_cpuload_0',
'gyro_bias[2]_estimator_sensor_bias_0',
'ev_hpos[1]_estimator_innovation_test_ratios_0',
'gyro_bias[0]_estimator_sensor_bias_0',
'ev_hvel[1]_estimator_innovation_test_ratios_0',
'ev_hpos[1]_estimator_innovation_variances_0',
'accel_bias[2]_estimator_sensor_bias_0',
]
def get_dataset(file_dir):
    local_path = file_dir + 'ICMP_attack'

    filenames = glob.glob(local_path + "/*.csv")

    li = []

    for filename in filenames:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    frame = frame[column_name]

    npframe = frame.to_numpy()
    d_icmp = np.zeros((int(frame.shape[0]/8),8,frame.shape[1]))
    y_icmp = []
    trainx = []
    trainy = []
    for i in range(int(frame.shape[0]/8)):
        d_icmp[i][:][:] = npframe[i*8:8+i*8][:]
        y_icmp.append(int(1))
        trainx.append(npframe[i*8:8+i*8][:])
        trainy.append(int(1))

    local_path = file_dir + 'FDI_attack'

    filenames = glob.glob(local_path + "/*.csv")

    li = []

    for filename in filenames:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    frame = frame[column_name]

    npframe = frame.to_numpy()
    d_fdi = np.zeros((int(frame.shape[0]/8),8,frame.shape[1]))
    y_fdi = []
    for i in range(int(frame.shape[0]/8)):
        d_fdi[i][:][:] = npframe[i*8:8+i*8][:]
        y_fdi.append(int(2))
        trainx.append(npframe[i*8:8+i*8][:])
        trainy.append(int(2))    

    local_path = file_dir + 'ICMP_no_attack'

    filenames = glob.glob(local_path + "/*.csv")

    li = []

    for filename in filenames:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    frame = frame[column_name]

    npframe = frame.to_numpy()
    d_icmp_no_attack = np.zeros((int(frame.shape[0]/8),8,frame.shape[1]))
    y_icmp_no_attack = []
    for i in range(int(frame.shape[0]/8)):
        d_icmp_no_attack[i][:][:] = npframe[i*8:8+i*8][:]
        y_icmp_no_attack.append(int(0))
        trainx.append(npframe[i*8:8+i*8][:])
        trainy.append(int(0))    
        
    local_path = file_dir + 'FDI_no_attack'

    filenames = glob.glob(local_path + "/*.csv")

    li = []

    for filename in filenames:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    frame = frame[column_name]

    npframe = frame.to_numpy()
    d_fdi_no_attack = np.zeros((int(frame.shape[0]/8),8,frame.shape[1]))
    y_fdi_no_attack = []
    for i in range(int(frame.shape[0]/8)):
        d_fdi_no_attack[i][:][:] = npframe[i*8:8+i*8][:]
        y_fdi_no_attack.append(int(0))
        trainx.append(npframe[i*8:8+i*8][:])
        trainy.append(int(0))    
        

    X_train, X_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.2)
    
    x_train_d = torch.tensor(np.array(X_train))
    y_train_d = torch.tensor(np.array(y_train), dtype = torch.long)

    x_test_d = torch.tensor(np.array(X_test))
    y_test_d = torch.tensor(np.array(y_test), dtype = torch.long)

    dtrainset = TensorDataset(x_train_d, y_train_d)
    dtestset = TensorDataset(x_test_d, y_test_d)
    return dtrainset,dtestset