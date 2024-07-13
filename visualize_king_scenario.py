import os
import sys
import cv2
import json
import glob
import math
import argparse
import csv
import pandas as pd
sys.path.append(os.getcwd())
from shapely.geometry.polygon import Polygon
import numpy as np
import carla
import pygame
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from PIL import ImageDraw, Image

# Global variables
GPU_PI = torch.tensor([np.pi], device="cuda", dtype=torch.float32)
BB_EXTENT = torch.tensor([[1.0,2.35],[1.0,-2.35],[-1.0, -2.35],[-1.0, 2.35]]).cuda()

PIXELS_PER_METER = 5

# Color
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

VEH_COLL_THRESH = 0.02

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class MapImage(object):
    def __init__(self, carla_world, carla_map, pixels_per_meter=10):
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0

        waypoints = carla_map.generate_waypoints(2)
        margin = 100#50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels))
        self.big_lane_surface = pygame.Surface((width_in_pixels, width_in_pixels))
        self.draw_road_map(
                self.big_map_surface, self.big_lane_surface,
                carla_world, carla_map, self.world_to_pixel, self.world_to_pixel_width)
        self.map_surface = self.big_map_surface
        self.lane_surface = self.big_lane_surface

    def draw_road_map(self, map_surface, lane_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        # map_surface.fill(COLOR_ALUMINIUM_4)
        map_surface.fill(COLOR_BLACK)
        precision = 0.05

        def draw_lane_marking(surface, points, solid=True):
            if solid:
                # pygame.draw.lines(surface, COLOR_ORANGE_0, False, points, 2)
                pygame.draw.lines(surface, COLOR_WHITE, False, points, 2)
            else:
                broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]
                for line in broken_lines:
                    # pygame.draw.lines(surface, COLOR_ORANGE_0, False, line, 2)
                    pygame.draw.lines(surface, COLOR_WHITE, False, line, 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            start = transform.location
            end = start + 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        start, end]], 4)
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        left, start, right]], 4)

        def draw_stop(surface, font_surface, transform, color=COLOR_ALUMINIUM_2):
            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = carla.Location(-forward_vector.y, forward_vector.x, forward_vector.z) * waypoint.lane_width/2 * 0.7

            line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                    (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]
            
            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

          

        def lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def does_cross_solid_line(waypoint, shift):
            w = carla_map.get_waypoint(lateral_shift(waypoint.transform, shift), project_to_road=False)
            if w is None or w.road_id != waypoint.road_id:
                return True
            else:
                return (w.lane_id * waypoint.lane_id < 0) or w.lane_id == waypoint.lane_id

        topology = [x[0] for x in carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
    
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)[0]

            left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

            polygon = left_marking + [x for x in reversed(right_marking)]
            polygon = [world_to_pixel(x) for x in polygon]

            if len(polygon) > 2:
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon, 10)
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon)

            if not waypoint.is_intersection:
                sample = waypoints[int(len(waypoints) / 2)]
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in left_marking],
                    does_cross_solid_line(sample, -sample.lane_width * 1.1))
                draw_lane_marking(
                    lane_surface,
                    [world_to_pixel(x) for x in right_marking],
                    does_cross_solid_line(sample, sample.lane_width * 1.1))
                    
                # Dian: Do not draw them arrows
                # for n, wp in enumerate(waypoints):
                #     if (n % 400) == 0:
                #         draw_arrow(map_surface, wp.transform)
        
        # actors = carla_world.get_actors()
        # stops_transform = [actor.get_transform() for actor in actors if 'stop' in actor.type_id]
        # font_size = world_to_pixel_width(1)            
        # font = pygame.font.SysFont('Arial', font_size, True)
        # font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        # font_surface = pygame.transform.scale(font_surface, (font_surface.get_width(),font_surface.get_height() * 2))
        
        # Dian: do not draw stop sign
        
        # for stop in stops_transform:
        #     draw_stop(map_surface,font_surface, stop)


    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))


class CarlaWrapper():
    def __init__(self, args):
        self._vehicle = None
        self.args = args
        self.town = None

    def _initialize_from_carla(self, town='Town01', port=2000):
        self.town = town
        self.client = carla.Client('localhost', port)

        self.client.set_timeout(10.0)

        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        map_image = MapImage(self.world, self.map, PIXELS_PER_METER)
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)
        lane = make_image(map_image.lane_surface)

        global_map = np.zeros((1, 4,) + road.shape)
        global_map[:, 0, ...] = road / 255.
        global_map[:, 1, ...] = lane / 255.

        global_map = torch.tensor(global_map, device=self.args.device, dtype=torch.float32)
        world_offset = torch.tensor(map_image._world_offset, device=self.args.device, dtype=torch.float32)

        return global_map, world_offset

class BaseRenderer():
    """
        Base class. Implements various common things, such as coordinate transforms,
        visualization and a function to render local views of the global map.
    """
    def __init__(self, args, map_offset, map_dims, viz=False):
        """
        """
        self.args = args
        self.map_offset = map_offset
        self.map_dims = map_dims

        if viz:
            self.PIXELS_AHEAD_VEHICLE = torch.tensor(
                [0],  # for visualization we center the ego vehicle
                device=self.args.device,
                dtype=torch.float32
            )
            #self.crop_dims = (800, 800)
            self.crop_dims = (300, 300)  # we use a bigger crop for visualization
        else:
            self.PIXELS_AHEAD_VEHICLE = torch.tensor(
                [100 + 10],
                device=self.args.device,
                dtype=torch.float32
            )
            self.crop_dims = (192, 192)

        self.gpu_pi = torch.tensor([np.pi], device=self.args.device, dtype=torch.float32)

        self.crop_scale = (
            self.crop_dims[1] / self.map_dims[1],
            self.crop_dims[0] / self.map_dims[0]
        )

        # we precompute several static quantities for efficiency
        self.world_to_rel_map_dims = torch.tensor(
            [self.map_dims[1],self.map_dims[0]],
            device=self.args.device,
            dtype=torch.float32
        )

        self.world_to_pix_crop_shift_tensor = torch.tensor(
            [0., - self.PIXELS_AHEAD_VEHICLE],
            device=self.args.device,
            dtype=torch.float32
        )

        self.world_to_pix_crop_half_crop_dims = torch.tensor(
            [self.crop_dims[1] / 2, self.crop_dims[0] / 2],
            device=self.args.device
        )

        self.get_local_birdview_scale_transform = torch.tensor(
            [[self.crop_scale[1], 0, 0],
            [0, self.crop_scale[0], 0],
            [0, 0, 1]],
            device=self.args.device,
            dtype=torch.float32,
        ).view(1, 3, 3).expand(self.args.batch_size, -1, -1)

        self.get_local_birdview_shift_tensor = torch.tensor(
            [0., - 2 * self.PIXELS_AHEAD_VEHICLE / self.map_dims[0]],
            device=self.args.device,
            dtype=torch.float32,
        )

    def get_local_birdview(self, global_map, position, orientation):
        """
        """
        global_map = global_map.expand(self.args.batch_size, -1, -1, -1)

        # convert position from world to relative image coordinates
        position = self.world_to_rel(position)
        orientation = orientation + self.gpu_pi / 2

        scale_transform = self.get_local_birdview_scale_transform

        zeros = torch.zeros_like(orientation)
        ones = torch.ones_like(orientation)


        rotation_transform = torch.stack(
            [torch.cos(orientation), -torch.sin(orientation), zeros,
             torch.sin(orientation),  torch.cos(orientation), zeros,
             zeros,                   zeros,                  ones],
             dim=-1,
        ).view(self.args.batch_size, 3, 3)

        shift = self.get_local_birdview_shift_tensor

        position = position + (rotation_transform[:, 0:2, 0:2] @ shift).unsqueeze(1)

        translation_transform = torch.stack(
            [ones,  zeros, position[..., 0:1] / self.crop_scale[0],\
             zeros, ones,  position[..., 1:2] / self.crop_scale[1],\
             zeros, zeros, ones],
            dim=-1,
        ).view(self.args.batch_size, 3, 3)

        # chain tansforms
        local_view_transform = scale_transform @ translation_transform @ rotation_transform

        affine_grid = F.affine_grid(
            local_view_transform[:, 0:2, :],
            (self.args.batch_size, 1, self.crop_dims[0], self.crop_dims[0]),
            align_corners=True,
        )

        # loop saves gpu memory
        local_views = []
        for batch_idx in range(self.args.batch_size):
            local_view_per_elem = F.grid_sample(
                global_map[batch_idx:batch_idx+1],
                affine_grid[batch_idx:batch_idx+1],
                align_corners=True,
            )
            local_views.append(local_view_per_elem)
        local_view = torch.cat(local_views, dim=0)

        return local_view

    def world_to_pix(self, pos):
        pos_px = (pos-self.map_offset) * PIXELS_PER_METER

        return pos_px

    def world_to_rel(self, pos):
        pos_px = self.world_to_pix(pos)
        pos_rel = pos_px / self.world_to_rel_map_dims

        pos_rel = pos_rel * 2 - 1

        return pos_rel

    def world_to_pix_crop(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):
        crop_yaw = crop_yaw + self.gpu_pi / 2
        batch_size = crop_pos.shape[0]

        # transform to crop pose
        rotation = torch.cat(
            [torch.cos(crop_yaw), -torch.sin(crop_yaw),
            torch.sin(crop_yaw),  torch.cos(crop_yaw)],
            dim=-1,
        ).view(batch_size, -1, 2, 2)

        crop_pos_px = self.world_to_pix(crop_pos)

        # correct for the fact that crop is only in front of ego agent
        shift = self.world_to_pix_crop_shift_tensor

        query_pos_px_map = self.world_to_pix(query_pos)

        query_pos_px = torch.transpose(rotation, -2, -1) @ \
            (query_pos_px_map - crop_pos_px).unsqueeze(-1)
        query_pos_px = query_pos_px.squeeze(-1) - shift

        # shift coordinate frame to top left corner of the crop
        pos_px_crop = query_pos_px + self.world_to_pix_crop_half_crop_dims

        return pos_px_crop

    def reset(self):
        """
        """
        pass

class BEVVisualizer:
    def __init__(self, args):
        """
        """
        self.args = args
        self.col_log_paths, self.init_log_paths = self.parse_scenario_log_dir()
        self.carla_wrapper = CarlaWrapper(args)


        self.town = None

    def visualize(self):
        """
        Visualizes logs in a simple abstract BEV representation that is centered
        on the ego agent. Can dump .gifs or .mp4s.
        """
        # loop over all logs
        folder = args.scenario_log_dir
        for scenario_name in sorted(os.listdir(folder)):
            scenario_name = scenario_name.split('.')[0]
            file_exist_flag = 0
            if file_exist_flag:
                continue
            # if scenario_name != '770':
            #     continue
            if os.path.isfile("output_gif/" + scenario_name + ".gif"): 
                file_exist_flag = 1
                continue
    #for init_log_path  in tqdm(
        #  self.col_log_paths, total=len(self.col_log_paths)
        # ):
        # dir = os.path.dirname(init_log_path)
        # if os.path.isfile(os.path.join(dir, "pixel_vis.gif")): folder + scenario_name + '/' + variant + "/" + scenario_name
        #     continue
            log = self.parse_csv_file(folder + scenario_name + '.csv')
            time = log.groupby("TIMESTAMP")
            states = []
            ego_states = []
            tp_states = []
            if scenario_name.split('_')[-1] == 'foward':
                scenario_name = re.sub('_foward', '', scenario_name)
            gt_tp = scenario_name.split('_')[-1]
            for name, group in time:
                pos = []
                yaw = []
                vel = []
                ego_pos = []
                ego_yaw = []
                ego_vel = []
                tp_pos = []
                tp_yaw = []
                tp_vel = []
                for  iter, data in group.iterrows():
                    if int(data['TRACK_ID']) == 123:
                        ego_pos.append([data["X"], data["Y"]])
                        ego_yaw.append(np.deg2rad([np.float32(data["YAW"])]))
                        ego_vel.append(data["V"])
                    elif int(data['TRACK_ID']) == int(gt_tp):
                        tp_pos.append([data["X"], data["Y"]])
                        tp_yaw.append(np.deg2rad([np.float32(data["YAW"])]))
                        tp_vel.append(data["V"])
                    else:
                        pos.append([data["X"], data["Y"]])
                        yaw.append(np.deg2rad([np.float32(data["YAW"])]))
                        vel.append(data["V"])
                states.append({'pos': pos, 'yaw': yaw, 'vel': vel})
                ego_states.append({'pos': ego_pos, 'yaw': ego_yaw, 'vel': ego_vel})
                tp_states.append({'pos': tp_pos, 'yaw': tp_yaw, 'vel': tp_vel})
            #print(states[0])
                

            # extract meta data
            town = log["CITY_NAME"][0]

    
            # set town in relevant components if necessary
            if town != self.town:
                global_map, map_offset = self.carla_wrapper._initialize_from_carla(town)
                renderer = BaseRenderer(
                    self.args, map_offset, global_map.shape[2:4], viz=True
                )

            bev_overview_vis_per_t = []

            for state, ego_state, tp_state in zip(states, ego_states, tp_states):
                for substate in ego_state:
                    ego_state[substate] = torch.tensor(
                            ego_state[substate],
                            device=self.args.device,
                    )
                #print(state["pos"].unsqueeze(0), state["pos"].unsqueeze(0)[:, 0:1])
                ego_pos_map = ego_state["pos"].unsqueeze(0)
                ego_pos_yaw = ego_state["yaw"].unsqueeze(0)
                # fetch local crop of map
                local_map = renderer.get_local_birdview(
                    global_map,
                    ego_pos_map[:, 0:1], # ego pos as origin
                    ego_pos_yaw[:, 0:1], # ego yaw as reference
                )

                ego_vehicle_corners = self.get_corners_vectorized(
                    BB_EXTENT,
                    ego_state["pos"].unsqueeze(0),
                    ego_state["yaw"].unsqueeze(0),
                )

                ego_vehicle_corners = renderer.world_to_pix_crop(
                        ego_vehicle_corners, 
                        ego_pos_map[:, 0:1], # ego pos as origin
                        ego_pos_yaw[:, 0:1], # ego yaw as reference
                )

                ego_vehicle_corners = ego_vehicle_corners.detach().cpu().numpy()
                ego_vehicle_corners = ego_vehicle_corners[0].reshape(ego_vehicle_corners.shape[1]//4,4,2)

                # for i in range(ego_vehicle_corners.shape[0]):
                #     bev_overview_vis_draw.polygon(ego_vehicle_corners[i].flatten(),fill=(105, 156, 219),outline=(0, 0, 0))
                #     bev_overview_vis_draw.polygon(np.concatenate([ego_vehicle_corners[i][2], ego_vehicle_corners[i][1], np.mean(ego_vehicle_corners[i], axis=0)]),outline=(0, 0, 0))
                ################################
                for substate in state:
                    state[substate] = torch.tensor(
                            state[substate],
                            device=self.args.device,
                    )

                vehicle_corners = self.get_corners_vectorized(
                    BB_EXTENT,
                    state["pos"].unsqueeze(0),
                    state["yaw"].unsqueeze(0),
                )

                vehicle_corners = renderer.world_to_pix_crop(
                        vehicle_corners, 
                        ego_pos_map[:, 0:1], # ego pos as origin
                        ego_pos_yaw[:, 0:1], # ego yaw as reference
                )

                vehicle_corners = vehicle_corners.detach().cpu().numpy()
                vehicle_corners = vehicle_corners[0].reshape(vehicle_corners.shape[1]//4,4,2)

                # for i in range(vehicle_corners.shape[0]):
                #     bev_overview_vis_draw.polygon(vehicle_corners[i].flatten(),fill=(222, 112, 97),outline=(0, 0, 0))
                #     bev_overview_vis_draw.polygon(np.concatenate([vehicle_corners[i][2], vehicle_corners[i][1], np.mean(vehicle_corners[i], axis=0)]),outline=(0, 0, 0))
                ####################################
                for substate in tp_state:
                    tp_state[substate] = torch.tensor(
                            tp_state[substate],
                            device=self.args.device,
                    )
                tp_vehicle_corners = self.get_corners_vectorized(
                    BB_EXTENT,
                    tp_state["pos"].unsqueeze(0),
                    tp_state["yaw"].unsqueeze(0),
                )

                tp_vehicle_corners = renderer.world_to_pix_crop(
                        tp_vehicle_corners, 
                        ego_pos_map[:, 0:1], # ego pos as origin
                        ego_pos_yaw[:, 0:1], # ego yaw as reference
                )

                tp_vehicle_corners = tp_vehicle_corners.detach().cpu().numpy()
                tp_vehicle_corners = tp_vehicle_corners[0].reshape(tp_vehicle_corners.shape[1]//4,4,2)

                bev_vis = self.tensor_to_pil(local_map)
                bev_overview_vis_draw = ImageDraw.Draw(bev_vis)

                for i in range(ego_vehicle_corners.shape[0]):
                    bev_overview_vis_draw.polygon(ego_vehicle_corners[i].flatten(),fill=(222, 112, 97),outline=(0, 0, 0))
                    bev_overview_vis_draw.polygon(np.concatenate([ego_vehicle_corners[i][2], ego_vehicle_corners[i][1], np.mean(ego_vehicle_corners[i], axis=0)]),outline=(0, 0, 0))
                
                for i in range(vehicle_corners.shape[0]):
                    bev_overview_vis_draw.polygon(vehicle_corners[i].flatten(),fill=(105, 156, 219),outline=(0, 0, 0))
                    bev_overview_vis_draw.polygon(np.concatenate([vehicle_corners[i][2], vehicle_corners[i][1], np.mean(vehicle_corners[i], axis=0)]),outline=(0, 0, 0))

                for i in range(tp_vehicle_corners.shape[0]):
                    bev_overview_vis_draw.polygon(tp_vehicle_corners[i].flatten(),fill=(255, 10, 10),outline=(0, 0, 0))
                    bev_overview_vis_draw.polygon(np.concatenate([tp_vehicle_corners[i][2], tp_vehicle_corners[i][1], np.mean(tp_vehicle_corners[i], axis=0)]),outline=(0, 0, 0))
                
                bev_overview_vis_per_t.append(bev_vis)

            # for t, state in enumerate(ego_states):
            #     # map dict of lists to dict of tensors
            #     for substate in state:
            #         state[substate] = torch.tensor(
            #                 state[substate],
            #                 device=self.args.device,
            #         )
            #     #print(state["pos"].unsqueeze(0), state["pos"].unsqueeze(0)[:, 0:1])
            #     ego_pos_map = state["pos"].unsqueeze(0)
            #     ego_pos_yaw = state["yaw"].unsqueeze(0)
            #     # fetch local crop of map
            #     local_map = renderer.get_local_birdview(
            #         global_map,
            #         ego_pos_map[:, 0:1], # ego pos as origin
            #         ego_pos_yaw[:, 0:1], # ego yaw as reference
            #     )

            #     vehicle_corners = self.get_corners_vectorized(
            #         BB_EXTENT,
            #         state["pos"].unsqueeze(0),
            #         state["yaw"].unsqueeze(0),
            #     )

            #     vehicle_corners = renderer.world_to_pix_crop(
            #             vehicle_corners, 
            #             ego_pos_map[:, 0:1], # ego pos as origin
            #             ego_pos_yaw[:, 0:1], # ego yaw as reference
            #     )

            #     vehicle_corners = vehicle_corners.detach().cpu().numpy()
            #     vehicle_corners = vehicle_corners[0].reshape(vehicle_corners.shape[1]//4,4,2)

            #     bev_vis = self.tensor_to_pil(local_map)
            #     bev_overview_vis_draw = ImageDraw.Draw(bev_vis)

            #     for i in range(vehicle_corners.shape[0]):
            #         bev_overview_vis_draw.polygon(vehicle_corners[i].flatten(),fill=(105, 156, 219),outline=(0, 0, 0))
            #         bev_overview_vis_draw.polygon(np.concatenate([vehicle_corners[i][2], vehicle_corners[i][1], np.mean(vehicle_corners[i], axis=0)]),outline=(0, 0, 0))

                

            #     bev_overview_vis_per_t.append(bev_vis)

            # for t, state in enumerate(states):
            #     # map dict of lists to dict of tensors
            #     for substate in state:
            #         state[substate] = torch.tensor(
            #                 state[substate],
            #                 device=self.args.device,
            #         )

            #     # fetch local crop of map
            #     local_map = renderer.get_local_birdview(
            #         global_map,
            #         ego_pos_map[:, 0:1], # ego pos as origin
            #         ego_pos_yaw[:, 0:1], # ego yaw as reference
            #     )

            #     vehicle_corners = self.get_corners_vectorized(
            #         BB_EXTENT,
            #         state["pos"].unsqueeze(0),
            #         state["yaw"].unsqueeze(0),
            #     )

            #     vehicle_corners = renderer.world_to_pix_crop(
            #             vehicle_corners, 
            #             ego_pos_map[:, 0:1], # ego pos as origin
            #             ego_pos_yaw[:, 0:1], # ego yaw as reference
            #     )

            #     vehicle_corners = vehicle_corners.detach().cpu().numpy()
            #     vehicle_corners = vehicle_corners[0].reshape(vehicle_corners.shape[1]//4,4,2)

            #     bev_vis = self.tensor_to_pil(local_map)
            #     bev_overview_vis_draw = ImageDraw.Draw(bev_vis)

            #     for i in range(vehicle_corners.shape[0]):
            #         bev_overview_vis_draw.polygon(vehicle_corners[i].flatten(),fill=(222, 112, 97),outline=(0, 0, 0))
            #         bev_overview_vis_draw.polygon(np.concatenate([vehicle_corners[i][2], vehicle_corners[i][1], np.mean(vehicle_corners[i], axis=0)]),outline=(0, 0, 0))

            #     bev_overview_vis_per_t.append(bev_vis)

            # for t, state in enumerate(tp_states):
            #     # map dict of lists to dict of tensors
            #     for substate in state:
            #         state[substate] = torch.tensor(
            #                 state[substate],
            #                 device=self.args.device,
            #         )

            #     # fetch local crop of map
            #     local_map = renderer.get_local_birdview(
            #         global_map,
            #         ego_pos_map[:, 0:1], # ego pos as origin
            #         ego_pos_yaw[:, 0:1], # ego yaw as reference
            #     )

            #     vehicle_corners = self.get_corners_vectorized(
            #         BB_EXTENT,
            #         state["pos"].unsqueeze(0),
            #         state["yaw"].unsqueeze(0),
            #     )

            #     vehicle_corners = renderer.world_to_pix_crop(
            #             vehicle_corners, 
            #             ego_pos_map[:, 0:1], # ego pos as origin
            #             ego_pos_yaw[:, 0:1], # ego yaw as reference
            #     )

            #     vehicle_corners = vehicle_corners.detach().cpu().numpy()
            #     vehicle_corners = vehicle_corners[0].reshape(vehicle_corners.shape[1]//4,4,2)

            #     bev_vis = self.tensor_to_pil(local_map)
            #     bev_overview_vis_draw = ImageDraw.Draw(bev_vis)

            #     for i in range(vehicle_corners.shape[0]):
            #         bev_overview_vis_draw.polygon(vehicle_corners[i].flatten(),fill=(255, 10, 10),outline=(0, 0, 0))
            #         bev_overview_vis_draw.polygon(np.concatenate([vehicle_corners[i][2], vehicle_corners[i][1], np.mean(vehicle_corners[i], axis=0)]),outline=(0, 0, 0))

            #     bev_overview_vis_per_t.append(bev_vis)
            #print(len(bev_overview_vis_per_t), bev_overview_vis_per_t)
            #save_path = os.path.join(os.path.dirname(init_log_path), "pixel_vis")
            save_path = os.path.join("output_gif/" + scenario_name)
            # save frames as gif
            bev_overview_vis_per_t[0].save(
                save_path + '.gif', 
                save_all=True, 
                append_images=bev_overview_vis_per_t[1:], 
                optimize=True,
                loop=0,
            )

            # save frames as .mp4
            # codec = cv2.VideoWriter_fourcc(*'mp4v')    
            # video_writer = cv2.VideoWriter(save_path + ".mp4",codec, 4, bev_overview_vis_per_t[0].size)       
            # for timestep in range(len(log["states"])):
            #         video_writer.write(cv2.cvtColor(
            #             np.array(bev_overview_vis_per_t[timestep]), cv2.COLOR_RGB2BGR))
            # video_writer.release()

    def tensor_to_pil(self, grid):
        """
        """
        colors = [
            (120, 120, 120), # road
            (253, 253, 17), # lane
            (0, 0, 142), # vehicle
        ]
        
        grid = grid.detach().cpu()

        grid_img = np.zeros((grid.shape[2:4] + (3,)), dtype=np.uint8)
        grid_img[...] = [225, 225, 225]
        
        for i in range(len(colors)):
            grid_img[grid[0, i, ...] > 0] = colors[i]

        pil_img = Image.fromarray(grid_img)

        return pil_img

    def get_corners_vectorized(self, extent, pos, yaw):
        yaw = GPU_PI/2 -yaw
        extent = extent.unsqueeze(-1)

        rot_mat = torch.cat(
            [
                torch.cos(yaw), torch.sin(yaw),
                -torch.sin(yaw), torch.cos(yaw),
            ],
            dim=-1,
        ).view(yaw.size(1), 1, 2, 2).expand(yaw.size(1), 4, 2, 2)

        rotated_corners = rot_mat @ extent

        rotated_corners = rotated_corners.view(yaw.size(1), 4, 2) + pos[0].unsqueeze(1)
        
        return rotated_corners.view(1, -1, 2)
    
    def parse_scenario_log_dir(self):
        """
        Parse generation results directory and gather 
        the JSON file paths from the per-route directories.
        """
        route_scenario_dirs = sorted(
            glob.glob(
                self.args.scenario_log_dir + "/RouteScenario*/", recursive=True
            ),
            key=lambda path: (path.split("_")[-6]),
        )

        label_file = "initial_scenario/scs_category_label.csv"
        #label_file = "./king_data/initial_scenario/agents_4/scs_category_label.csv"
        collision_type_name = ["junction_crossing", "LTAP", "TCD_violation", "lane_change", "opposite_direction", "rear_end"]
        # gather all records and results JSON files
        records_files = []
        init_scenario_records = []
        for dir in sorted(route_scenario_dirs):
            print(dir)
            collision_type_list = []
            with open(label_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == dir[-25:-1]:
                        for i in range(1, len(row)):
                            if row[i] == '1':
                                collision_type_list.append(collision_type_name[i-1])
            for type in collision_type_list:
                records_files.extend(
                    sorted(
                        glob.glob(dir + type + "/" + "/**/result_traj.csv")
                    )
                )
            init_scenario_records.extend(
                sorted(
                    glob.glob(dir + "initial/result_traj.csv")
                )
            )
        return records_files, init_scenario_records

    def parse_csv_file(self, records_file):
        """
        read csv file and return a list of dicts
        :param records_file: path to csv file
        :return: list of dicts
        """

        return (pd.read_csv(records_file))
def cal_front_direction_ego(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        # angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        angle = np.arccos(np.dot(v1_u, v2_u))        
        if math.isnan(angle):
            return 0.0
        else:
            return angle
def cal_front_direction(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        # angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if v1_u[1] < 0:
            v1_u = v1_u * (-1)
        angle = np.arccos(np.dot(v1_u, v2_u))
        #print("v1_u:", v1_u, "v2_u:", v2_u, "dot:", np.dot([-0.7, 0.7], v2_u), "angle:", np.arccos(np.dot([-0.7, 0.7], v2_u)))
        #print("v1_u:", v1_u, "v2_u:", v2_u, "dot:", np.dot([-0.7, -0.7], v2_u), "angle:", np.arccos(np.dot([-0.7, -0.7], v2_u)))
        # y<0, means 180~360
        
        if math.isnan(angle):
            return 0.0
        else:
            return angle
def angle_vectors(v1, v2):
    """ Returns angle between two vectors.  """
    # 若是車輛靜止不動 ego_vec為[0 0]
    if v1[0] < 0.0001 and v1[1] < 0.0001:
        v1_u = [1.0, 0.1]
    else:
        v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # 因為np.arccos給的範圍只有0~pi (180度)，但若cos值為-0.5，不論是120度或是240度，都會回傳120度，因此考慮三四象限的情況，強制轉180度到一二象限(因車輛和行人面積的對稱性，故可這麼做)
    #if v1_u[1] < 0:
    #    v1_u = v1_u * (-1)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if math.isnan(angle):
        return 0.0
    else:
        return angle
def txt_to_csv(args):
    folder = args.scenario_log_dir
    # route_scenario_dirs = sorted(
    #     glob.glob(
    #         args.scenario_log_dir + "/*/", recursive=True
    #     ),
    #     key=lambda path: (path.split("_")[-6]),
    # )
    # label_file = "initial_scenario/scs_category_label.csv"
    # #label_file = "./king_data/initial_scenario/agents_4/scs_category_label.csv"
    # collision_type_name = ["junction_crossing", "LTAP", "TCD_violation", "lane_change", "opposite_direction", "rear_end"]
    # # gather all records and results JSON files
    # records_files = []
    # init_scenario_records = []
    # for dir in sorted(route_scenario_dirs):
    #     print(dir)
    #     collision_type_list = []
    #     with open(label_file, 'r') as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             if row[0] == dir[-25:-1]:
    #                 for i in range(1, len(row)):
    #                     if row[i] == '1':
    #                         collision_type_list.append(collision_type_name[i-1])
    #     for type in collision_type_list:
    #         records_files.extend(
    #             sorted(
    #                 glob.glob(dir + type + "/" + "/**/result_traj.csv")
    #             )
    #         )
    #     init_scenario_records.extend(
    #         sorted(
    #             glob.glob(dir + "initial/result_traj.csv")
    #         )
    #     )
    
    for scenario_name in sorted(os.listdir(folder)):
        #if scenario_name != '112' and scenario_name != '170':
        #        continue
        for variant in sorted(os.listdir(folder + scenario_name + '/')):
            if int(scenario_name.split('-')[0]) > 100:
                with open('initial_scenario/agents_4/RouteScenario_' + scenario_name + '_to_' + scenario_name + '/init_scenario_records.json') as f:
                    log = json.load(f)
                    f.close()
                town = log["meta_data"]["town"]
            else:
                town_num = scenario_name.split('-')[0]
                if town_num == '10':
                    town = 'Town10HD'
                else:
                    town = 'Town0' + town_num
            plot_df = pd.DataFrame()
            vehicle_list = []

            ego_df = pd.read_csv(os.path.join(
                'scripts/datasets_carla/all/initial_scenario_test_all_frames/' + scenario_name + '_' + variant + '.txt'), sep='\t', header=None)
            ego_list = []
            for track_id, remain_df in ego_df.groupby(1):
                if int(track_id) == 123:
                    for f_index, frame_id in enumerate(remain_df[0].values):
                        if f_index == len(remain_df[0].values) - 1:
                            break
                        x = remain_df[2].values[f_index]
                        y = remain_df[3].values[f_index]
                        x_next = remain_df[2].values[f_index + 1]
                        y_next = remain_df[3].values[f_index + 1]
                        ego_vec = [x_next - x, y_next - y]
                        v = math.sqrt(ego_vec[0] ** 2 + ego_vec[1] ** 2) / 0.1
                        #angle = np.rad2deg(cal_front_direction_ego(ego_vec, [1, 0]))
                        angle = remain_df[4].values[f_index]
                        vehicle_list.append([remain_df[0].values[f_index + 1], track_id, 'vehicle', x_next, y_next, v, angle, town])

            for filename in sorted(os.listdir(folder + scenario_name + '/' + variant + '/')):
                print(scenario_name, variant, filename)
                if filename == 'result_traj.csv':
                    continue
                if filename == 'prediction':
                    continue
                if filename == 'collsion_description':
                    continue
                if filename.split('.')[-1] == 'gif':
                    continue
                
                traj_df = pd.read_csv(os.path.join(
                        folder + scenario_name + '/' + variant + '/', filename), sep='\t', header=None)
                
                if args.input_frames == 1:
                    if filename.split('-')[0] == '00':
                        for track_id, id_df in traj_df.groupby(1):
                            if int(track_id) == 123:
                                continue
                            for frame_id in id_df[0].values:
                                if frame_id < args.input_frames:
                                    x = id_df[2].values[frame_id]
                                    y = id_df[3].values[frame_id]
                                    x_next = id_df[2].values[frame_id + 1]
                                    y_next = id_df[3].values[frame_id + 1]
                                    ego_vec = [x_next - x, y_next - y]
                                    v = math.sqrt(ego_vec[0] ** 2 + ego_vec[1] ** 2) / 0.1
                                    #angle = np.rad2deg(cal_front_direction(ego_vec, [1, 0]))
                                    angle = id_df[4].values[frame_id]
                                    #print(v, angle)
                                    vehicle_list.append([id_df[0].values[frame_id + 1], track_id, 'vehicle', x_next, y_next, v, angle, town])
                                    #for frame_, remain_df in id_df.groupby(0):
                                    #    if int(frame_) <= 4:
                                    #        print(remain_df)
                    else:
                        skip = 1
                        for track_id, id_df in traj_df.groupby(1):
                            if int(track_id) == 123:
                                continue
                            for f_index, frame_id in enumerate(id_df[0].values):
                                if f_index == 3:
                                    x = id_df[2].values[f_index]
                                    y = id_df[3].values[f_index]
                                    x_next = id_df[2].values[f_index + 1 * skip]
                                    y_next = id_df[3].values[f_index + 1 * skip]
                                    #ego_vec = [y_next - y,x_next - x]
                                    #ego_vec = [x_next * (-1) - x * (-1), y_next - y]
                                    #ego_vec = [x_next - x, y_next * (-1) - y * (-1)]
                                    ego_vec = [x_next - x, y_next - y]
                                    #angle = np.rad2deg(angle_vectors(ego_vec, [1, 0]))
                                    v = math.sqrt(ego_vec[0] ** 2 + ego_vec[1] ** 2) / (0.1 * skip)
                                    #angle = np.rad2deg(cal_front_direction(ego_vec, [1, 0]))
                                    angle = id_df[4].values[f_index]
                                    
                                    vehicle_list.append([id_df[0].values[f_index + 1 * skip], track_id, 'vehicle', x_next, y_next, v, angle, town])
                else:
                    now_file_num = int(filename.split('-')[0])
                    if now_file_num > 3:
                        continue
                    skip = 1
                    for track_id, id_df in traj_df.groupby(1):
                        if int(track_id) == 123:
                            continue
                        for f_index, frame_id in enumerate(id_df[0].values):
                            if f_index < args.output_frames:
                                if f_index + 1 + args.output_frames * now_file_num >= 40:
                                    continue
                                x = id_df[2].values[f_index]
                                y = id_df[3].values[f_index]
                                x_next = id_df[2].values[f_index + 1 * skip]
                                y_next = id_df[3].values[f_index + 1 * skip]
                                #ego_vec = [y_next - y,x_next - x]
                                #ego_vec = [x_next * (-1) - x * (-1), y_next - y]
                                #ego_vec = [x_next - x, y_next * (-1) - y * (-1)]
                                ego_vec = [x_next - x, y_next - y]
                                #angle = np.rad2deg(angle_vectors(ego_vec, [1, 0]))
                                v = math.sqrt(ego_vec[0] ** 2 + ego_vec[1] ** 2) / (0.1 * skip)
                                angle = np.rad2deg(cal_front_direction(ego_vec, [1, 0]))
                                #angle = id_df[4].values[f_index]
                                
                                vehicle_list.append([f_index + 1 + args.output_frames * now_file_num, track_id, 'vehicle', x_next, y_next, v, angle, town])

            #print(pd.DataFrame(vehicle_list))
            pd.DataFrame(vehicle_list, columns =['TIMESTAMP', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 'V', 'YAW', 'CITY_NAME']).to_csv(folder + scenario_name + '/' + variant + '/result_traj.csv', index=False)

def metric(args):
    #all_cnt = {"junction_crossing": 0, "LTAP": 0, "TCD_violation": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    all_cnt = {"junction_crossing": 0, "LTAP": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    col_cnt = {"junction_crossing": 0, "LTAP": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    inside_cnt = {"junction_crossing": 0, "LTAP": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    distance_cnt = {"junction_crossing": 0, "LTAP": 0, "lane_change": 0, "opposite_direction": 0, "rear_end": 0}
    vehicle_length = 4.7
    vehicle_width = 2
    all_data_num = 0
    col_data_num = 0
    yaw_offset_degree = 30
    folder = args.scenario_log_dir
    for scenario_name in sorted(os.listdir(folder)):
        # if scenario_name == '5-i-1' or scenario_name == '5-s-7':
        #     continue
        # if scenario_name != '114':
        #        continue
        # if scenario_name != '137':
        #        continue
        for variant in sorted(os.listdir(folder + scenario_name + '/')):
            #print(scenario_name, variant)
            collision_flag = 0
            sav_path = folder + scenario_name + \
                '/' + variant + '/prediction'
            if not os.path.exists(sav_path):
                os.makedirs(sav_path)
            if variant == 'lane_change' or variant == 'TCD_violation':
                ideal_yaw_distance = 15
            elif variant == 'junction_crossing' or variant == 'LTAP':
                ideal_yaw_distance = 90
            elif variant == 'opposite_direction':
                ideal_yaw_distance = 180
            elif variant == 'rear_end':
                ideal_yaw_distance = 0
            all_cnt[variant] += 1
            all_data_num += 1
            vehicle_list = []
            traj_df = pd.read_csv(folder + scenario_name + '/' + variant + '/result_traj.csv')
            for track_id, remain_df in traj_df.groupby('TRACK_ID'):
                vehicle_list.append(remain_df)
            ego_list = []
            for track_id, remain_df in traj_df.groupby('TRACK_ID'):
                if int(track_id) == int(123):
                    ego_list.append(remain_df)
            scenario_length = len(vehicle_list[0])
            if scenario_name == '5-i-1' or scenario_name == '5-s-7':
                scenario_length -= 10
            for t in range(1, scenario_length):
                ego_x = ego_list[0].loc[t - 1, 'X']
                ego_x_next = ego_list[0].loc[t, 'X']
                ego_y = ego_list[0].loc[t - 1, 'Y']
                ego_y_next = ego_list[0].loc[t, 'Y']
                ego_vec = [ego_y_next - ego_y,
                                    ego_x_next * (-1) - ego_x * (-1)]
                ego_angle = np.rad2deg(
                             angle_vectors(ego_vec, [1, 0])) * np.pi / 180
                ego_rec = [ego_x_next, ego_y_next, vehicle_width
                                                , vehicle_length, ego_angle]
                x_1 = float(np.cos(
                    ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                x_2 = float(np.cos(
                    ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                x_3 = float(np.cos(
                    ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                x_4 = float(np.cos(
                    ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                y_1 = float(np.sin(
                    ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                y_2 = float(np.sin(
                    ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                y_3 = float(np.sin(
                    ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                y_4 = float(np.sin(
                    ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                ego_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
                # plt.plot([x_1, x_2, x_4, x_3, x_1], [
                #                             y_1, y_2, y_4, y_3, y_1], '-',  color='lime', markersize=3)
                for n in range(len(vehicle_list)):
                    vl = vehicle_list[n].to_numpy()
                    # vl : frame, id, x, y
                    now_id = vl[0][1]
                    if int(now_id) == 123:
                        continue
                    real_pred_x = vl[t - 1][3]
                    real_pred_x_next = vl[t][3]
                    real_pred_y = vl[t - 1][4]
                    real_pred_y_next = vl[t][4]
                    other_vec = [real_pred_y_next - real_pred_y,
                                         real_pred_x_next * (-1) - real_pred_x * (-1)]
                    other_angle = np.rad2deg(
                             angle_vectors(other_vec, [1, 0])) * np.pi / 180
                    # other_angle = vl[past_len][4]
                    # ego_angle = ego_list[0][4][int(filename_t) + past_len]
                    #print(ego_x, ego_y, real_pred_x, real_pred_y)
                    ego_rec = [real_pred_x_next, real_pred_y_next, vehicle_width
                                                , vehicle_length, other_angle]
                    x_1 = float(np.cos(
                        ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                    x_2 = float(np.cos(
                        ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[0])
                    x_3 = float(np.cos(
                        ego_rec[4])*(-ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                    x_4 = float(np.cos(
                        ego_rec[4])*(ego_rec[2]/2) - np.sin(ego_rec[4])*(ego_rec[3]/2) + ego_rec[0])
                    y_1 = float(np.sin(
                        ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                    y_2 = float(np.sin(
                        ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(-ego_rec[3]/2) + ego_rec[1])
                    y_3 = float(np.sin(
                        ego_rec[4])*(-ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                    y_4 = float(np.sin(
                        ego_rec[4])*(ego_rec[2]/2) + np.cos(ego_rec[4])*(ego_rec[3]/2) + ego_rec[1])
                    other_polygon = Polygon([[x_1,y_1], [x_2,y_2], [x_4,y_4], [x_3,y_3], [x_1,y_1]])
                    # plt.plot([x_1, x_2, x_4, x_3, x_1], [
                    #                         y_1, y_2, y_4, y_3, y_1], '-',  color='black', markersize=3)
                    
                    cur_iou = ego_polygon.intersection(other_polygon).area / ego_polygon.union(other_polygon).area
                    if cur_iou > VEH_COLL_THRESH:
                        collision_flag = 1
                        col_data_num += 1
                        col_cnt[variant] += 1
                        real_yaw_distance = abs(abs(ego_angle * 180 / np.pi) - abs(other_angle * 180 / np.pi))
                        distance = abs(abs(real_yaw_distance) - abs(ideal_yaw_distance))
                        distance_cnt[variant] += distance
                        #print("Collide!", distance, " distance")
                        if distance <= yaw_offset_degree:
                            print(scenario_name, variant)
                            inside_cnt[variant] += 1
                    if collision_flag:
                        break
                if collision_flag:
                    break
                # plt.xlim(ego_list[0].loc[30, 'X'] - 75,
                #             ego_list[0].loc[30, 'X'] + 75)
                # plt.ylim(ego_list[0].loc[30, 'Y'] - 75,
                #             ego_list[0].loc[30, 'Y'] + 75)
                # plt.savefig(sav_path + '/' + str(t) + '.png', dpi=500)
                # plt.close()
    print(all_cnt)
    # for variant_key in all_cnt:
    #     distance_cnt[variant_key] /= all_cnt[variant_key]
    all_df = pd.DataFrame(all_cnt, index=['all'])
    col_df = pd.DataFrame(col_cnt, index=['collision_scenarios'])
    cr_df = pd.DataFrame(col_cnt, index=['collision_rate'])
    cr_df = col_df.copy()
    for variant_key in cr_df:
        cr_df[variant_key] = col_cnt[variant_key] / all_cnt[variant_key]
    inside_df = pd.DataFrame(inside_cnt, index=['similarity(inside)'])
    ir_df = inside_df.copy()
    for variant_key in ir_df:
        if int(col_cnt[variant_key]) == 0:
            ir_df[variant_key] = 0
            distance_cnt[variant_key] = 0
        else:
            ir_df[variant_key] = inside_cnt[variant_key] / col_cnt[variant_key]
            distance_cnt[variant_key] /= col_cnt[variant_key]
    distance_df = pd.DataFrame(distance_cnt, index=['similarity(distance)'])
    result = pd.concat([all_df, col_df, cr_df, inside_df, ir_df, distance_df]).T
    result.columns = ['all', 'collision_scenarios', 'collision_rate', 'similarity(inside)', 'similarity(inside rate)', 'similarity(distance)']
    result.to_csv('close_loop_8f_metric.csv')


def main(args):
    """
    """
    if args.from_txt:
      txt_to_csv(args)
    vizualizer =  BEVVisualizer(args)
    vizualizer.visualize()
    #metric(args)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_frames",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--output_frames",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--scenario_log_dir",
        type=str,
        # default="output_csv_moving_foward/",
        default="output_csv/",
        help="The directory containing the per-route directories with the "
             "corresponding scenario log .json files.",
    )
    parser.add_argument(
        "--opt_iter",
        type=int,
        default=-1,
        help="Specifies at which iteration in the optimization process the "
             "scenarios should be visualized. Set to -1 to automatically "
             "select the critical perturbation for each scenario.",
    )
    parser.add_argument(
        "--optim_method",
        default="Adam",
        choices=["Adam", "Both_Paths"]
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="Carla port."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of parallel simulations."
    )
    parser.add_argument(
        "--from_txt",
        type=int,
        default=0,
        help="The number of parallel simulations."
    )

    args = parser.parse_args()

    main(args)
