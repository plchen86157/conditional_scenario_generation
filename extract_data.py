import pandas as pd
import numpy as np
import os
from collections import defaultdict
import shutil
import sys
if __name__ == "__main__":
    data_path = '/home/yoyo/sgan/agents_1_attacker'
    #data_path = '/home/yoyo/sgan/king_agents_4/'
    output_path = 'carla_all_data/'
    train_split = 0.8
    val_split = 0.1
    collsion_cnt = {"junction_crossing": 0, "LTAP": 0, "TCD_violation": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    now_cnt = {"junction_crossing": 0, "LTAP": 0, "TCD_violation": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    for scenario in sorted(os.listdir(data_path)):
        for scenario_type in sorted(os.listdir(data_path + '/' + scenario)):
            for variant_id in sorted(os.listdir(data_path + '/' + scenario + '/' + scenario_type)):
                collsion_cnt[scenario_type] += 1
    print(collsion_cnt)
    for scenario in sorted(os.listdir(data_path)):
        print(scenario)
        route_num = scenario.split('_')[-1]
        town_num = route_num.split('-')[0]
        for scenario_type in sorted(os.listdir(data_path + '/' + scenario)):            
            for variant_id in sorted(os.listdir(data_path + '/' + scenario + '/' + scenario_type)):
                now_cnt[scenario_type] += 1
                
                sce_df = pd.read_csv(data_path + '/' + scenario + '/' + scenario_type + '/' + variant_id + '/' + 'result_traj.csv')
                temp_pd = pd.DataFrame()
                frame_list = []
                for frame, remain_df_frame in sce_df.groupby("TIMESTAMP"):
                    if int(frame * 10) % 2 == 0:
                        frame_list.append(frame)
                        temp_pd = pd.concat([temp_pd, remain_df_frame], axis=0)
                #vehicle_pd = pd.DataFrame()
                #for object_type, remain_df in temp_pd.groupby("OBJECT_TYPE"):
                #    vehicle_pd = pd.concat([vehicle_pd, remain_df], axis=0)
                sce_df = temp_pd
                if len(frame_list) < 20:
                    continue
                #############################
                # after data trimming, start frame has changed
                # topology_time = str(int(np.unique(temp_pd['TIMESTAMP'].values)[7] * 10))
                # topo_source_path = "/home/yoyo/sgan/initial_scenario/agents_4/" + "RouteScenario_" + route_num + "_to_" + route_num + "/topology_150x150/" + topology_time + '.npy'
                # topo_destination_path = output_path + 'topology_150x150/' + route_num + '_' + scenario_type + '_' + variant_id + '.npy'
                # shutil.copyfile(topo_source_path, topo_destination_path)
                sce_df.loc[sce_df.TRACK_ID == 123, 'OBJECT_TYPE'] = 'AGENT'
                sce_df.loc[sce_df.TRACK_ID != 123, 'OBJECT_TYPE'] = 'vehicle'
                if town_num == '10':
                    sce_df.loc[sce_df.TRACK_ID != -1, 'CITY_NAME'] = 'Town10HD'
                    sce_df.rename(columns={'Yaw':'YAW'}, inplace=True)
                elif int(town_num) < 10:
                    sce_df.loc[sce_df.TRACK_ID != -1, 'CITY_NAME'] = 'Town0' + town_num
                    sce_df.rename(columns={'Yaw':'YAW'}, inplace=True)
                if now_cnt[scenario_type] < collsion_cnt[scenario_type] * train_split:
                    traj_path = output_path + 'trajectory/train/' + route_num + '_' + scenario_type + '_' + variant_id + '.csv'
                elif collsion_cnt[scenario_type] * train_split < now_cnt[scenario_type] < collsion_cnt[scenario_type] * (train_split + val_split):
                    traj_path = output_path + 'trajectory/val/' + route_num + '_' + scenario_type + '_' + variant_id + '.csv'
                else:
                    traj_path = output_path + 'trajectory/test_obs/' + route_num + '_' + scenario_type + '_' + variant_id + '.csv'
                sce_df.to_csv(traj_path, index=False)
                topo_source_path = data_path + '/' + scenario + '/' + scenario_type + '/' + variant_id + '/' + 'topology.npy'
                topo_destination_path = output_path + 'topology/' + route_num + '_' + scenario_type + '_' + variant_id + '.npy'
                shutil.copyfile(topo_source_path, topo_destination_path)
