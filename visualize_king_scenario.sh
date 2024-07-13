#!/bin/bash

export CARLA_ROOT=../../Desktop/CARLA_0.9.13 # path to your carla root
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$(pwd -P)/leaderboard
export PYTHONPATH=$PYTHONPATH:$(pwd -P)/scenario_runner

SERVICE="CarlaUE4"
../../Desktop/CARLA_0.9.13/CarlaUE4.sh &
sleep 5
# for filename in king_data/initial_scenario/agents_4/Route*;
# do
#     if pgrep "$SERVICE" >/dev/null
#     then
#         echo "$SERVICE is running"
#     else
#         echo "$SERVICE is  stopped"
#         ../../CarlaUE4.sh & sleep 5	
#     fi
#     route_id=${filename: -3}
#     echo $route_id
#     python scenario_initializer_v2.py \
#         --task 4 \
#         --tp_type 4 \
#         --update True \
#         --playback Collect_data \
#         --route_id $route_id
#     echo -e "\n"
#     echo "$route_id Finish Collect Data"
#     pgrep $SERVICE | xargs kill -9
#     sleep 5
# done

for ((i=0;i<100;i++))
do
    if pgrep "$SERVICE" >/dev/null
    then
        echo "$SERVICE is running"
    else
        echo "$SERVICE is  stopped"
        ../../Desktop/CARLA_0.9.13/CarlaUE4.sh & sleep 5	
    fi
    python visualize_king_scenario.py
    #python visualize_king_scenario.py
    pgrep $SERVICE | xargs kill -9
    sleep 5
done