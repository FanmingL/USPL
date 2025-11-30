# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation
from scipy.ndimage import distance_transform_edt


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.stone_height = np.zeros((cfg.num_rows, cfg.num_cols, 4))
        self.num_goals = cfg.num_goals

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            if hasattr(cfg, "max_difficulty"):
                self.curiculum(random=True, max_difficulty=cfg.max_difficulty)
            else:
                self.curiculum(random=True)
            # self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
                half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
                structure = np.ones((half_edge_width*2+1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                if self.cfg.num_rows == 1:
                    difficulty = 1.0
                else:
                    difficulty = i / (self.cfg.num_rows-1)
                choice = j / self.cfg.num_cols + 0.001
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(choice, np.random.uniform(0.7, 1), max_difficulty=True)
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    terrain = self.make_terrain(choice, difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    def make_terrain(self, choice, difficulty, max_difficulty=False):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.02 + 0.14 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            idx = 0
            if choice < self.proportions[0]/ 2:
                idx = 1
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # self.add_roughness(terrain)
        elif choice < self.proportions[2]:
            idx = 2
            if choice<self.proportions[1]:
                idx = 3
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[4]:
            idx = 4
            if choice<self.proportions[3]:
                idx = 5
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[5]:
            idx = 6
            num_rectangles = 20
            rectangle_min_size = 0.5
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[6]:
            idx = 7
            stones_size = 1.5 - 1.2*difficulty
            # terrain_utils.stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)
            half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            stepping_stones_terrain(terrain, stone_size=1.5-0.2*difficulty, stone_distance=0.0+0.4*difficulty, max_height=0.2*difficulty, platform_size=1.2)
            self.add_roughness(terrain)
        elif choice < self.proportions[7]:
            idx = 8
            # gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            gap_parkour_terrain(terrain, difficulty, platform_size=4)
            self.add_roughness(terrain)
        elif choice < self.proportions[8]:
            idx = 9
            self.add_roughness(terrain)
            # pass
        elif choice < self.proportions[9]:
            idx = 10
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[10]:
            idx = 11
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty *= 1.3
                if not self.cfg.no_flat:
                    difficulty -= 0.1
                if difficulty > 1:
                    half_slope_difficulty = 1.0
                elif difficulty < 0:
                    self.add_roughness(terrain)
                    terrain.slope_vector = np.array([1, 0., 0]).astype(np.float32)
                    return terrain
                else:
                    half_slope_difficulty = difficulty
            wall_width = 4 - half_slope_difficulty * 4
            # terrain_utils.wall_terrain(terrain, height=1, start2center=0.7)
            # terrain_utils.tanh_terrain(terrain, height=1.0, start2center=0.7)
            if self.cfg.flat_wall:
                half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            else:
                half_sloped_terrain(terrain, wall_width=wall_width, start2center=0.5, max_height=1.5)
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height
        elif choice < self.proportions[11]:
            idx = 12
            # half platform terrain
            half_platform_terrain(terrain, max_height=0.1 + 0.4 * difficulty )
            self.add_roughness(terrain, difficulty=1)
        elif choice < self.proportions[13]:
            idx = 13
            height = 0.1 + 0.3 * difficulty
            if choice < self.proportions[12]:
                idx = 14
                height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=1., step_height=height, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[14]:
            x_range = [-0.1, 0.1+0.3*difficulty]  # offset to stone_len
            y_range = [0.2, 0.3+0.1*difficulty]
            stone_len = [0.9 - 0.3*difficulty, 1 - 0.2*difficulty]#2 * round((0.6) / 2.0, 1)
            incline_height = 0.25*difficulty
            last_incline_height = incline_height + 0.1 - 0.1*difficulty
            parkour_terrain(terrain,
                            num_stones=self.num_goals - 2,
                            x_range=x_range, 
                            y_range=y_range,
                            incline_height=incline_height,
                            stone_len=stone_len,
                            stone_width=1.0, 
                            last_incline_height=last_incline_height,
                            pad_height=0,
                            pit_depth=[0.2, 1])
            idx = 15
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[15]:
            idx = 16
            # parkour_hurdle_terrain(terrain,
            #                        num_stones=self.num_goals - 2,
            #                        stone_len=0.1+0.3*difficulty,
            #                        hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.25*difficulty],
            #                        pad_height=0,
            #                        x_range=[1.2, 2.2],
            #                        y_range=self.cfg.y_range,
            #                        half_valid_width=[0.4, 0.8],
            #                        )
            if max_difficulty:
                max_difficulty = False
                parkour_middle_choose_terrain_paper(terrain,
                                              num_stones=self.num_goals - 2,
                                              target_radius_coeff=difficulty,
                                              hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty],
                                              pad_height=-0.8,
                                              y_range=self.cfg.y_range,
                                              half_valid_width=[0.45, 1],
                                              pad_width=1.5,
                                              flat=True,
                                              terrain_length=self.cfg.terrain_length,
                                              length=self.cfg.middle_choose_length,
                                              max_difficulty=max_difficulty,
                                              )
            else:
                parkour_middle_choose_terrain(terrain,
                                       num_stones=self.num_goals - 2,
                                       target_radius_coeff=difficulty,
                                       hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty],
                                       pad_height=-0.8,
                                       y_range=self.cfg.y_range,
                                       half_valid_width=[0.45, 1],
                                       pad_width=1.5,
                                       flat=True,
                                       terrain_length=self.cfg.terrain_length,
                                       length=self.cfg.middle_choose_length,
                                      max_difficulty=max_difficulty,
                                       )

            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[16]:
            idx = 17
            parkour_circle_plat_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   target_radius_coeff=difficulty,
                                   hurdle_height_range=[0.1+0.1*difficulty, 0.15+0.15*difficulty],
                                   pad_height=-0.8,
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.45, 1],
                                   flat=True,
                                   terrain_length=self.cfg.terrain_length,
                                   )
            self.add_roughness(terrain)
        elif choice < self.proportions[17]:
            idx = 18
            # parkour_step_terrain(terrain,
            #                        num_stones=self.num_goals - 2,
            #                        step_height=0.1 + 0.35*difficulty,
            #                        x_range=[0.3,1.5],
            #                        y_range=self.cfg.y_range,
            #                        half_valid_width=[0.5, 1],
            #                        pad_height=0,
            #                        )
            parkour_vertical_line_plat_terrain(terrain,
                                        num_stones=self.num_goals - 2,
                                        target_radius_coeff=difficulty,
                                        hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty],
                                        pad_height=-0.8,
                                        y_range=self.cfg.y_range,
                                        half_valid_width=[0.45, 1],
                                       pad_width=1.5,
                                       # pad_width=0.1,
                                       # pad_height=0.5,
                                        flat=True,
                                       terrain_length=self.cfg.terrain_length,
                                        )
            self.add_roughness(terrain)
        elif choice < self.proportions[18]:
            idx = 19
            # parkour_gap_terrain(terrain,
            #                     num_gaps=self.num_goals - 2,
            #                     gap_size=0.1 + 0.7 * difficulty,
            #                     gap_depth=[0.2, 1],
            #                     pad_height=0,
            #                     x_range=[0.8, 1.5],
            #                     y_range=self.cfg.y_range,
            #                     half_valid_width=[0.6, 1.2],
            #                     # flat=True
            #                     )
            # parkour_square_terrain(terrain,
            #                        num_stones=self.num_goals - 2,
            #                        target_radius_coeff=difficulty,
            #                        hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty],
            #                        pad_height=-0.8,
            #                        y_range=self.cfg.y_range,
            #                        half_valid_width=[0.45, 1],
            #                        pad_width=1.5,
            #                        flat=True,
            #                        terrain_length=self.cfg.terrain_length,
            #                        )
            parkour_S_line_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   target_radius_coeff=difficulty,
                                   hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty],
                                   pad_height=-0.8,
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.45, 1],
                                   pad_width=1.5,
                                   # pad_width=0.1,
                                   # pad_height=0.5,
                                   flat=True,
                                   terrain_length=self.cfg.terrain_length,
                                   )
            self.add_roughness(terrain)
            # self.add_roughness(terrain)
        elif choice < self.proportions[19]:
            idx = 20
            if max_difficulty:
                max_difficulty = False
                parkour_square_terrain_paper(terrain,
                                       num_stones=self.num_goals - 2,
                                       target_radius_coeff=difficulty,
                                       hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty],
                                       pad_height=-0.8,
                                       y_range=self.cfg.y_range,
                                       half_valid_width=[0.45, 1],
                                       pad_width=1.5,
                                       flat=True,
                                       terrain_length=self.cfg.terrain_length,
                                       length=2.1,
                                       max_difficulty=max_difficulty
                                       )
            else:
                parkour_square_terrain(terrain,
                                             num_stones=self.num_goals - 2,
                                             target_radius_coeff=difficulty,
                                             hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty],
                                             pad_height=-0.8,
                                             y_range=self.cfg.y_range,
                                             half_valid_width=[0.45, 1],
                                             pad_width=1.5,
                                             flat=True,
                                             terrain_length=self.cfg.terrain_length,
                                             length=2.1,
                                             max_difficulty=max_difficulty
                                             )

            self.add_roughness(terrain)

        # np.set_printoptions(precision=2)
        # print(np.array(self.proportions), choice)
        terrain.idx = idx
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale) # within 1 meter square range
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx
        self.goals[i, j, :, :2] = terrain.goals + [i * self.env_length, j * self.env_width]
        if hasattr(terrain, 'stone_heights'):
            self.stone_height[i, j, :] = terrain.stone_heights
        # self.env_slope_vec[i, j] = terrain.slope_vector

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def gap_parkour_terrain(terrain, difficulty, platform_size=2.):
    gap_size = 0.1 + 0.3 * difficulty
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -400
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    slope_angle = 0.1 + difficulty * 1
    offset = 1 + 9 * difficulty#10
    scale = 15
    wall_center_x = [center_x - x1, center_x, center_x + x1]
    wall_center_y = [center_y - y1, center_y, center_y + y1]

    # for i in range(center_y + y1, center_y + y2):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)
    
    # for i in range(center_y - y2, center_y - y1):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)

    # for i in range(center_x + x1, center_x + x2):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)
    
    # for i in range(center_x - x2, center_x - x1):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)

def parkour_terrain(terrain, 
                    platform_len=2.5, 
                    platform_height=0., 
                    num_stones=8, 
                    x_range=[1.8, 1.9], 
                    y_range=[0., 0.1], 
                    z_range=[-0.2, 0.2],
                    stone_len=1.0,
                    stone_width=0.6,
                    pad_width=0.1,
                    pad_height=0.5,
                    incline_height=0.1,
                    last_incline_height=0.6,
                    last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones+2, 2))
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    
    mid_y = terrain.length // 2  # length is actually y width
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len -  stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    # dis_z = np.random.randint(dis_z_min, dis_z_max)
    dis_z = 0
    
    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2*(left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
        
        goals[i+1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height
    
def parkour_gap_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_gaps=8,
                           gap_size=0.3,
                           x_range=[1.6, 2.4],
                           y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           gap_depth=-200,
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_gaps+2, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    
    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth
    
    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, 
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_hurdle_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                           stone_len=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
            terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def parkour_circle_plat_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           target_radius_coeff=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           terrain_length=16.0,
                           flat=False):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200

    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height
    # pad_radius = (terrain_length / 2) * 0.8 # 6.5
    # pad_radius = (terrain_length / 2) * 0.68 # 6.5
    pad_radius = (terrain_length / 2) * 0.75 # 6.5

    target_radius = np.random.uniform(0.4, 0.85) * pad_radius
    target_angle = np.random.uniform(0, 2 * np.pi)
    # if np.random.rand() < 0.1:
    #     target_angle = np.random.randint(0, 2) * np.pi
    target_x = target_radius * np.cos(target_angle)
    target_y = target_radius * np.sin(target_angle)

    goal_x = (target_radius - 0.5) * np.cos(target_angle)
    goal_y = (target_radius - 0.5) * np.sin(target_angle)

    # stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)

    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len + int(terrain_length / 2 / terrain.horizontal_scale)
    goals[0] = [dis_x - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 2.0 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    target_x = round(target_x / terrain.horizontal_scale)
    target_y = round(target_y / terrain.horizontal_scale)
    half_bound = round(0.2 / terrain.horizontal_scale)

    # # 获取地形的宽和高
    height, width = terrain.height_field_raw.shape

    # 中心点
    center_x = width // 2
    center_y = height // 2
    # TODO: comment
    # terrain.height_field_raw[center_y + target_x - half_bound:center_y +target_x + half_bound, center_x + target_y - half_bound:center_x +  target_y + half_bound] = int(-0.5 // terrain.vertical_scale)

    goal_x = round(goal_x / terrain.horizontal_scale)
    goal_y = round(goal_y / terrain.horizontal_scale)

    goal_x = center_y + goal_x
    goal_y = center_x + goal_y

    goals[-1] = [goal_x, goal_y]
    terrain.goals = goals * terrain.horizontal_scale
    # pad edges

    pad_height = int(pad_height // terrain.vertical_scale)

    # 创建坐标网格
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))

    # 计算到中心的距离
    distances = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2) * terrain.horizontal_scale

    # 找到超出半径的区域
    mask = (distances > pad_radius) #  & ((np.abs(x_indices - center_x) * terrain.horizontal_scale) > 0.5)

    # 应用掩码设置高度
    terrain.height_field_raw[mask] = pad_height

    # triangle_
    triangle_center_x_1 = center_x
    triangle_center_y_1 = center_y + round(pad_radius // terrain.horizontal_scale) + round(0.2 // terrain.horizontal_scale)
    vec_ab = np.array([goal_y - triangle_center_x_1, goal_x - triangle_center_y_1])
    distance = np.linalg.norm(vec_ab) * terrain.horizontal_scale
    rel_distance = distance / terrain_length * 1.5 + 0.5
    vec_ab_unit = vec_ab / np.linalg.norm(vec_ab)
    vec_perp_unit = np.array([-vec_ab_unit[1], vec_ab_unit[0]])
    l = round(1.6 * rel_distance  / terrain.horizontal_scale)
    m = round(0.3 / terrain.horizontal_scale)
    # 计算矩形的四个顶点
    half_l = l / 2
    half_m = m / 2
    A = np.array([triangle_center_x_1, triangle_center_y_1])
    corner1 = np.array(A) + half_l * vec_ab_unit + half_m * vec_perp_unit
    corner2 = np.array(A) + half_l * vec_ab_unit - half_m * vec_perp_unit
    corner3 = np.array(A) - half_l * vec_ab_unit - half_m * vec_perp_unit
    corner4 = np.array(A) - half_l * vec_ab_unit + half_m * vec_perp_unit
    corners = np.array([corner1, corner2, corner3, corner4])
    points = np.stack([x_indices, y_indices], axis=-1)  # 所有点坐标，形状 (H, W, 2)
    # 定义叉积计算函数
    def cross_product(v1, v2):
        return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    # 计算点到矩形每条边的向量
    vec1 = points - corners[0]  # 点到 corner1 的向量
    vec2 = points - corners[1]
    vec3 = points - corners[2]
    vec4 = points - corners[3]

    edge1 = corners[1] - corners[0]  # 边 corner1 -> corner2
    edge2 = corners[2] - corners[1]  # 边 corner2 -> corner3
    edge3 = corners[3] - corners[2]  # 边 corner3 -> corner4
    edge4 = corners[0] - corners[3]  # 边 corner4 -> corner1

    # 计算每个点与四条边的叉积
    cross1 = cross_product(edge1, vec1)
    cross2 = cross_product(edge2, vec2)
    cross3 = cross_product(edge3, vec3)
    cross4 = cross_product(edge4, vec4)

    # 判断每个点是否在矩形内部
    mask = (cross1 >= 0) & (cross2 >= 0) & (cross3 >= 0) & (cross4 >= 0)
    mask = mask | ((cross1 <= 0) & (cross2 <= 0) & (cross3 <= 0) & (cross4 <= 0))
    terrain.height_field_raw[mask] = round(0.5 / terrain.vertical_scale)
    #
    # def create_rectangle_mask(size, A, B, l, m):
    #
    #
    #     # 创建坐标网格
    #     x, y = np.meshgrid(np.arange(W), np.arange(H))  # x 和 y 的形状是 (H, W)
    #     points = np.stack([x, y], axis=-1)  # 所有点坐标，形状 (H, W, 2)
    #
    #     # 定义叉积计算函数
    #     def cross_product(v1, v2):
    #         return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    #
    #     # 计算点到矩形每条边的向量
    #     vec1 = points - corners[0]  # 点到 corner1 的向量
    #     vec2 = points - corners[1]
    #     vec3 = points - corners[2]
    #     vec4 = points - corners[3]
    #
    #     edge1 = corners[1] - corners[0]  # 边 corner1 -> corner2
    #     edge2 = corners[2] - corners[1]  # 边 corner2 -> corner3
    #     edge3 = corners[3] - corners[2]  # 边 corner3 -> corner4
    #     edge4 = corners[0] - corners[3]  # 边 corner4 -> corner1
    #
    #     # 计算每个点与四条边的叉积
    #     cross1 = cross_product(edge1, vec1)
    #     cross2 = cross_product(edge2, vec2)
    #     cross3 = cross_product(edge3, vec3)
    #     cross4 = cross_product(edge4, vec4)
    #
    #     # 判断每个点是否在矩形内部
    #     mask = (cross1 >= 0) & (cross2 >= 0) & (cross3 >= 0) & (cross4 >= 0)
    #
    #     return mask.astype(int)


def parkour_square_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           target_radius_coeff=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           length=1.8,
                           terrain_length=16.0,
                           flat=False,
                           max_difficulty=False):
    goals = np.zeros((num_stones + 2, 2))
    height, width = terrain.height_field_raw.shape
    center_x = width // 2
    center_y = height // 2
    terrain.height_field_raw[:] = int(pad_height // terrain.vertical_scale)
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    w = 0.4
    mask = ((np.abs(x_indices - center_x) * terrain.horizontal_scale) < length) & (
                (np.abs(y_indices - center_y) * terrain.horizontal_scale) < length)

    # 应用掩码设置高度
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)

    cord = length - w - 0.1 # 1.6
    corners = [(0, (cord, cord)), (1, (-cord, -cord)), (2, (cord, -cord)), (3, (-cord, cord))]
    np.random.shuffle(corners)
    height_0 = np.random.uniform(0.0, 0.5)  # 0.05 ~ 0.5           # 0.05
    height_1 = height_0 + np.random.uniform(0.1, 0.25)  # 0.1 ~ 0.75  # 0.15
    height_2 = height_1 + np.random.uniform(0.1, 0.15)  # 0.2 ~ 0.9   # 0.25
    height_3 = 0.0

    heights = [height_0, height_1, height_2, height_3]
    heights_in_order = [(item0[0], item) for item0, item in zip(corners, heights)]
    heights_in_order = sorted(heights_in_order, key=lambda x: x[0])
    heights_in_order = [item[1] for item in heights_in_order]
    for idx, (corner, height) in enumerate(zip(corners, heights)):
        corner = corner[1]
        corner = (np.sign(corner[0]) * (np.abs(corner[0]) + w), np.sign(corner[1]) * (np.abs(corner[1]) + w))
        target_x = corner[0] / terrain.horizontal_scale + center_x
        target_y = corner[1] / terrain.horizontal_scale + center_y
        mask = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) +
                (np.abs(y_indices - target_y) * terrain.horizontal_scale) < 1.0)
        if idx == 1:
            mask_plat = terrain.height_field_raw < int(-0.75 / terrain.vertical_scale)
            mask2 = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) +
                    (np.abs(y_indices - target_y) * terrain.horizontal_scale) < 2.2)
            terrain.height_field_raw[mask2 & mask_plat] = int(-0.52 / terrain.vertical_scale)
            mask2 = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) +
                     (np.abs(y_indices - target_y) * terrain.horizontal_scale) < 1.6)
            terrain.height_field_raw[mask2 & mask_plat] = int(-0.26 / terrain.vertical_scale)
            mask3 = ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                     ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) < (length * 3.0))
            lll = np.random.uniform(1.75, 2.1) if not max_difficulty else 1.75
            mask3 = mask3 & ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                            ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) > (length * lll))
            mask3 = mask3 & (terrain.height_field_raw > int(-0.75 / terrain.vertical_scale))
            terrain.height_field_raw[mask3] = int(-0.8 / terrain.vertical_scale)
            terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)
            #
            # mask3 = ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
            #          ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) < (length * 2.45))
            # mask3 = mask3 & ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
            #                 ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) > (length * 2.1))
            # mask3 = mask3 & (terrain.height_field_raw > int(-0.75 / terrain.vertical_scale))
            # terrain.height_field_raw[mask3] = int(-0.8 / terrain.vertical_scale)
        else:
            terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)


        mask = ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                 ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) < (length * 2.45))
        # mask = mask & ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
        #                 ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) > (length * 1.95))
        # mask = mask & (terrain.height_field_raw > int(-0.75 / terrain.vertical_scale))
        # terrain.height_field_raw[mask] = int(-0.6 / terrain.vertical_scale)
        mask = (np.abs(x_indices - center_x) * terrain.horizontal_scale < 0.1) & (
                np.abs(y_indices - center_y) * terrain.horizontal_scale < 3.0
        )
        mask = mask | (np.abs(x_indices - center_x) * terrain.horizontal_scale < 3.0) & (
                np.abs(y_indices - center_y) * terrain.horizontal_scale < 0.1
        )
        # terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)
        if idx == 1:
            goals[-1] = [corner[1] / terrain.horizontal_scale + center_y,
                         corner[0] / terrain.horizontal_scale + center_x]
            # corner2 = (
            #     np.sign(corner[0]) * (np.abs(corner[0]) + 2.2), np.sign(corner[1]) * (np.abs(corner[1]) + 2.2))
            # target_x2 = corner2[0] / terrain.horizontal_scale + center_x
            # target_y2 = corner2[1] / terrain.horizontal_scale + center_y
            # mask = (((np.abs(x_indices - target_x2) * terrain.horizontal_scale) < 0.25) &
            #         ((np.abs(y_indices - target_y2) * terrain.horizontal_scale) < 0.25))
            # terrain.height_field_raw[mask] = int(0.2 / terrain.vertical_scale)
        elif idx < 3:
            # corner2 = (
            #     np.sign(corner[0]) * (np.abs(corner[0]) + 2.2), np.sign(corner[1]) * (np.abs(corner[1]) + 2.2))
            # target_x2 = corner2[0] / terrain.horizontal_scale + center_x
            # target_y2 = corner2[1] / terrain.horizontal_scale + center_y
            # mask = (((np.abs(x_indices - target_x2) * terrain.horizontal_scale) < 0.25) &
            #         ((np.abs(y_indices - target_y2) * terrain.horizontal_scale) < 0.25))
            # terrain.height_field_raw[mask] = int(-0.5 / terrain.vertical_scale)
            pass
    terrain.goals = goals * terrain.horizontal_scale
    mask = (np.abs(((x_indices - center_x) * terrain.horizontal_scale) -
                   ((y_indices - center_y) * terrain.horizontal_scale)) < 0.6)
    mask = mask | (np.abs(((x_indices - center_x) * terrain.horizontal_scale) +
                          ((y_indices - center_y) * terrain.horizontal_scale)) < 0.6)
    mask = mask & (terrain.height_field_raw >= int(0.05 / terrain.vertical_scale))
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)
    terrain.stone_heights = np.array(heights_in_order)
    return heights_in_order


def parkour_square_terrain_paper(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           target_radius_coeff=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           length=1.8,
                           terrain_length=16.0,
                           flat=False,
                           max_difficulty=False):
    goals = np.zeros((num_stones + 2, 2))
    height, width = terrain.height_field_raw.shape
    center_x = width // 2
    center_y = height // 2
    terrain.height_field_raw[:] = int(pad_height // terrain.vertical_scale)
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    w = 0.4
    mask = ((np.abs(x_indices - center_x) * terrain.horizontal_scale) < length) & (
                (np.abs(y_indices - center_y) * terrain.horizontal_scale) < length)

    # 应用掩码设置高度
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)

    cord = length - w - 0.1 # 1.6
    corners = [(0, (cord, cord)), (1, (-cord, -cord)), (2, (cord, -cord)), (3, (-cord, cord))]
    np.random.shuffle(corners)
    height_0 = np.random.uniform(0.0, 0.5)  # 0.05 ~ 0.5           # 0.05
    height_1 = height_0 + np.random.uniform(0.1, 0.25)  # 0.1 ~ 0.75  # 0.15
    height_2 = height_1 + np.random.uniform(0.1, 0.15)  # 0.2 ~ 0.9   # 0.25
    height_3 = 0.0
    heights = [height_0, height_1, height_2, height_3]
    print(f'heights: {heights} ')

    heights_in_order = [(item0[0], item) for item0, item in zip(corners, heights)]
    heights_in_order = sorted(heights_in_order, key=lambda x: x[0])
    heights_in_order = [item[1] for item in heights_in_order]
    print(f'heights_in_order: {[_ for _ in zip(corners, heights)]} ')
    corners = [(0, (1.6, -1.6)),(1, (-1.6, -1.6)), (2, (1.6, 1.6)), (3, (-1.6, 1.6))]
    heights = [0.5, 0.63, 0.75, 0.0]
    for idx, (corner, height) in enumerate(zip(corners, heights)):
        corner = corner[1]
        corner = (np.sign(corner[0]) * (np.abs(corner[0]) + w), np.sign(corner[1]) * (np.abs(corner[1]) + w))
        target_x = corner[0] / terrain.horizontal_scale + center_x
        target_y = corner[1] / terrain.horizontal_scale + center_y
        mask = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) +
                (np.abs(y_indices - target_y) * terrain.horizontal_scale) < 1.0)
        if idx == 1:
            mask_plat = terrain.height_field_raw < int(-0.75 / terrain.vertical_scale)
            mask2 = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) +
                    (np.abs(y_indices - target_y) * terrain.horizontal_scale) < 2.2)
            terrain.height_field_raw[mask2 & mask_plat] = int(-0.52 / terrain.vertical_scale)
            mask2 = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) +
                     (np.abs(y_indices - target_y) * terrain.horizontal_scale) < 1.6)
            terrain.height_field_raw[mask2 & mask_plat] = int(-0.26 / terrain.vertical_scale)
            mask3 = ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                     ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) < (length * 3.0))
            lll = np.random.uniform(1.75, 2.1) if not max_difficulty else 1.75
            lll = 2.0
            mask3 = mask3 & ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                            ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) > (length * lll))
            mask3 = mask3 & (terrain.height_field_raw > int(-0.75 / terrain.vertical_scale))
            terrain.height_field_raw[mask3] = int(-0.8 / terrain.vertical_scale)
            terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)
            #
            # mask3 = ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
            #          ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) < (length * 2.45))
            # mask3 = mask3 & ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
            #                 ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) > (length * 2.1))
            # mask3 = mask3 & (terrain.height_field_raw > int(-0.75 / terrain.vertical_scale))
            # terrain.height_field_raw[mask3] = int(-0.8 / terrain.vertical_scale)
        else:
            terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)


        mask = ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                 ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) < (length * 2.45))
        # mask = mask & ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
        #                 ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) > (length * 1.95))
        # mask = mask & (terrain.height_field_raw > int(-0.75 / terrain.vertical_scale))
        # terrain.height_field_raw[mask] = int(-0.6 / terrain.vertical_scale)
        mask = (np.abs(x_indices - center_x) * terrain.horizontal_scale < 0.1) & (
                np.abs(y_indices - center_y) * terrain.horizontal_scale < 3.0
        )
        mask = mask | (np.abs(x_indices - center_x) * terrain.horizontal_scale < 3.0) & (
                np.abs(y_indices - center_y) * terrain.horizontal_scale < 0.1
        )
        # terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)
        if idx == 1:
            goals[-1] = [corner[1] / terrain.horizontal_scale + center_y,
                         corner[0] / terrain.horizontal_scale + center_x]
            # corner2 = (
            #     np.sign(corner[0]) * (np.abs(corner[0]) + 2.2), np.sign(corner[1]) * (np.abs(corner[1]) + 2.2))
            # target_x2 = corner2[0] / terrain.horizontal_scale + center_x
            # target_y2 = corner2[1] / terrain.horizontal_scale + center_y
            # mask = (((np.abs(x_indices - target_x2) * terrain.horizontal_scale) < 0.25) &
            #         ((np.abs(y_indices - target_y2) * terrain.horizontal_scale) < 0.25))
            # terrain.height_field_raw[mask] = int(0.2 / terrain.vertical_scale)
        elif idx < 3:
            # corner2 = (
            #     np.sign(corner[0]) * (np.abs(corner[0]) + 2.2), np.sign(corner[1]) * (np.abs(corner[1]) + 2.2))
            # target_x2 = corner2[0] / terrain.horizontal_scale + center_x
            # target_y2 = corner2[1] / terrain.horizontal_scale + center_y
            # mask = (((np.abs(x_indices - target_x2) * terrain.horizontal_scale) < 0.25) &
            #         ((np.abs(y_indices - target_y2) * terrain.horizontal_scale) < 0.25))
            # terrain.height_field_raw[mask] = int(-0.5 / terrain.vertical_scale)
            pass
    terrain.goals = goals * terrain.horizontal_scale
    mask = (np.abs(((x_indices - center_x) * terrain.horizontal_scale) -
                   ((y_indices - center_y) * terrain.horizontal_scale)) < 0.6)
    mask = mask | (np.abs(((x_indices - center_x) * terrain.horizontal_scale) +
                          ((y_indices - center_y) * terrain.horizontal_scale)) < 0.6)
    mask = mask & (terrain.height_field_raw >= int(0.05 / terrain.vertical_scale))
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)
    terrain.stone_heights = np.array(heights_in_order)
    # mask_3 = (np.abs(((x_indices - center_x) * terrain.horizontal_scale) -
    #                ((y_indices - center_y) * terrain.horizontal_scale)) > 1.0)
    mask_3 = terrain.height_field_raw > int(-0.6 / terrain.vertical_scale)
    dist_to_region = ndimage.distance_transform_edt(~mask_3)
    # 满足距离条件的位置设为 True，其余为 False
    mask_3 = dist_to_region >= 10
    sss = terrain.height_field_raw[mask_3]
    rnd = np.random.uniform(-0.05, 0.05, sss.shape)
    terrain.height_field_raw[mask_3] = np.round((rnd - 0.8) / terrain.vertical_scale)
    return heights_in_order

def parkour_square_terrain_old(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           target_radius_coeff=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           terrain_length=16.0,
                           flat=False):
    goals = np.zeros((num_stones + 2, 2))
    height, width = terrain.height_field_raw.shape
    center_x = width // 2
    center_y = height // 2
    terrain.height_field_raw[:] = int(pad_height // terrain.vertical_scale)
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    # 找到超出半径的区域
    length = 1.25
    mask = ((np.abs(x_indices - center_x) * terrain.horizontal_scale) < length) & ((np.abs(y_indices - center_y) * terrain.horizontal_scale) < length)
    mask_center = mask
    # 应用掩码设置高度
    terrain.height_field_raw[mask] = int(0.4 / terrain.vertical_scale)
    mask_boarder_left_right = ((np.abs(x_indices - center_x) * terrain.horizontal_scale) >= length) & ((np.abs(x_indices - center_x) * terrain.horizontal_scale) < (length+0.5)) & ((np.abs(y_indices - center_y) * terrain.horizontal_scale) < (0.45 + length))
    mask_boarder_up_down = ((np.abs(y_indices - center_y) * terrain.horizontal_scale) >= length) & (
                (np.abs(y_indices - center_y) * terrain.horizontal_scale) < (length + 0.5)) & (
                                          (np.abs(x_indices - center_x) * terrain.horizontal_scale) < (length + 0.45))
    mask_boarder = mask_boarder_left_right | mask_boarder_up_down

    mask_boarder_left_right = ((np.abs(x_indices - center_x) * terrain.horizontal_scale) >= (length+0.5)) & (
                (np.abs(x_indices - center_x) * terrain.horizontal_scale) < (length + 0.6)) & (
                                          (np.abs(y_indices - center_y) * terrain.horizontal_scale) < (length - 0.1))
    mask_boarder_up_down = ((np.abs(y_indices - center_y) * terrain.horizontal_scale) >= (length + 0.5)) & (
            (np.abs(y_indices - center_y) * terrain.horizontal_scale) < (length + 0.6)) & (
                                   (np.abs(x_indices - center_x) * terrain.horizontal_scale) < (length-0.1))
    mask_boarder2 = mask_boarder_left_right | mask_boarder_up_down

    target_attach = np.random.randint(0, 4)
    boarder_2 = 0.7
    mmius = 0.2
    if target_attach == 0:
        target_x = length + 1 + 0.5 - boarder_2 - mmius
        target_y = length + 1 + 0.5 - boarder_2 - mmius
    elif target_attach == 1:
        target_x = -(length + 1 + 0.5 - boarder_2 - mmius)
        target_y = length + 1 + 0.5 - boarder_2 - mmius
    elif target_attach == 2:
        target_x = (length + 1 + 0.5 - boarder_2) - mmius
        target_y = -(length + 1 + 0.5 - boarder_2 - mmius)
    elif target_attach == 3:
        target_x = -(length + 1 + 0.5 - boarder_2 - mmius)
        target_y = -(length + 1 + 0.5 - boarder_2 - mmius)
    # if target_attach == 0:
    #     target_x = 1.5 + 1 + 0.5 - boarder_2
    #     target_y = np.random.uniform(-1.0, 1.0)
    # elif target_attach == 1:
    #     target_x = np.random.uniform(-1.0, 1.0)
    #     target_y = 1.5 + 1 + 0.5  - boarder_2
    # elif target_attach == 2:
    #     target_x = -(1.5 + 1 + 0.5 - boarder_2)
    #     target_y = np.random.uniform(-1.0, 1.0)
    # elif target_attach == 3:
    #     target_x = np.random.uniform(-1.0, 1.0)
    #     target_y = -(1.5 + 1 + 0.5 - boarder_2)
    target_x_original = target_x / terrain.horizontal_scale
    target_y_original = target_y / terrain.horizontal_scale
    square_center_x = np.sign(target_x) * (np.abs(target_x) - 0.25) / terrain.horizontal_scale + center_x
    square_center_y = np.sign(target_y) * (np.abs(target_y) - 0.25) / terrain.horizontal_scale + center_y

    target_x = target_x / terrain.horizontal_scale + center_x
    target_y = target_y / terrain.horizontal_scale + center_y

    # goals[-1] = [target_x, target_y]
    # all_mask = mask | mask_boarder
    # distance = distance_transform_edt(~all_mask)
    # 选择距离不超过N且原值为-0.8的点
    # selected = (distance <= 0.65 / 4 / terrain.horizontal_scale) & (~all_mask)
    # 将这些点的值设为0.4
    # terrain.height_field_raw[selected] = round(0.8 / terrain.vertical_scale)
    goals[-1] = [target_y, target_x]
    boarder_3 = 0.6
    mask = ((np.abs(x_indices - square_center_x) * terrain.horizontal_scale) < boarder_3) & (
                (np.abs(y_indices - square_center_y) * terrain.horizontal_scale) < boarder_3)
    terrain.height_field_raw[mask] = int(0.15 / terrain.vertical_scale)

    # boarder_4 = 0.5
    # target_x_2 = center_x - target_x_original
    # target_y_2 = center_y - target_y_original
    # mask2 = ((np.abs(x_indices - target_x_2) * terrain.horizontal_scale) < boarder_4) & (
    #         (np.abs(y_indices - target_y_2) * terrain.horizontal_scale) < boarder_4)
    # target_x_2 = center_x + target_x_original
    # target_y_2 = center_y - target_y_original
    # mask2 = mask2 | ((np.abs(x_indices - target_x_2) * terrain.horizontal_scale) < boarder_4) & (
    #         (np.abs(y_indices - target_y_2) * terrain.horizontal_scale) < boarder_4)
    # target_x_2 = center_x - target_x_original
    # target_y_2 = center_y + target_y_original
    # mask2 = mask2 | ((np.abs(x_indices - target_x_2) * terrain.horizontal_scale) < boarder_4) & (
    #         (np.abs(y_indices - target_y_2) * terrain.horizontal_scale) < boarder_4)
    # mask2 = mask2 | ((np.abs(x_indices - target_x) * terrain.horizontal_scale) < boarder_4) & (
    #         (np.abs(y_indices - target_y) * terrain.horizontal_scale) < boarder_4)

    # mask2 = mask2 & selected

    # terrain.height_field_raw[mask2] = int(0.4 // terrain.vertical_scale)
    terrain.goals = goals * terrain.horizontal_scale
    stair_height = 0.4
    terrain.height_field_raw[mask_boarder | mask_center] = int(stair_height / terrain.vertical_scale)
    terrain.height_field_raw[mask_boarder2] = int(0.7 / terrain.vertical_scale)
    # terrain.height_field_raw[mask] = int(0.4 / terrain.vertical_scale)


def parkour_middle_choose_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           target_radius_coeff=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           length=1.8,
                           terrain_length=16.0,
                           flat=False,
                           max_difficulty=False):
    goals = np.zeros((num_stones + 2, 2))
    height, width = terrain.height_field_raw.shape
    center_x = width // 2
    center_y = height // 2
    terrain.height_field_raw[:] = int(pad_height // terrain.vertical_scale)
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    # 找到超出半径的区域
    # length = np.random.uniform(2.0, 3.0)
    # length = np.random.uniform(2.0, 2.4)
    # length = np.random.uniform(2.0, 2.2)
    # length = 1.8
    # length = np.random.uniform(2.9, 3.0)
    # length = np.random.uniform(2.5, 3.0)
    w = 0.4
    mask = ((np.abs(x_indices - center_x) * terrain.horizontal_scale) < length) & ((np.abs(y_indices - center_y) * terrain.horizontal_scale) < length)

    # 应用掩码设置高度
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)

    cord = length - w - 0.1
    corners = [(0, (cord, cord)), (1, (-cord, -cord)), (2, (cord, -cord)), (3, (-cord, cord))]
    np.random.shuffle(corners)
    # corners.append((3, (-cord, cord)))
    # difficulty_level = np.random.uniform(0, 1)
    height_0 = np.random.uniform(0.0, 0.5)  # 0.00 ~ 0.5           # 0.05
    height_1 = height_0 + np.random.uniform(0.1, 0.25)  # 0.1 ~ 0.75  # 0.15
    height_2 = height_1 + np.random.uniform(0.1, 0.15)  # 0.2 ~ 0.9   # 0.25
    # height_0 = np.random.uniform(0.05, 0.5)  # 0.05 ~ 0.5           # 0.05
    # height_1 = height_0 + np.random.uniform(0.1, 0.101 + 0.15 * difficulty_level)  # 0.1 ~ 0.75  # 0.15
    # height_2 = height_1 + np.random.uniform(0.1, 0.101 + 0.05 * difficulty_level)  # 0.2 ~ 0.9   # 0.25
    height_3 = 0.0
    if max_difficulty:
        height_0 = 0.25
        height_1 = 0.37
        height_2 = 0.49
        height_3 = 0.0
    # if np.random.rand() < 0.85:
    #     # height_0 = np.random.uniform(0.1, 0.2) # 0.1 ~ 0.2
    #     # height_1 = height_0 + np.random.uniform(0.1, 0.15) # 0.2 ~ 0.35
    #     # height_2 = height_1 + np.random.uniform(0.1, 0.15) # 0.3 ~ 0.5
    #     # height_3 = 0.0
    #
    # else:
    #     height_0 = np.random.uniform(-0.15, -0.05)
    #     height_1 = np.random.uniform(0.1, 0.2)
    #     height_2 = height_1 + np.random.uniform(0.1, 0.15)
    #     height_3 = 0.0
    heights = [height_0, height_1, height_2, height_3]
    heights_in_order = [(item0[0], item) for item0, item in zip(corners, heights)]
    heights_in_order = sorted(heights_in_order, key=lambda x: x[0])
    heights_in_order = [item[1] for item in heights_in_order]
    for idx, (corner, height) in enumerate(zip(corners, heights)):
        corner = corner[1]
        corner = (np.sign(corner[0]) * (np.abs(corner[0]) + w), np.sign(corner[1]) * (np.abs(corner[1]) + w))
        target_x = corner[0] / terrain.horizontal_scale + center_x
        target_y = corner[1] / terrain.horizontal_scale + center_y
        mask = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) +
                (np.abs(y_indices - target_y) * terrain.horizontal_scale) < 1.0)
        terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)

        mask = ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) < (length * 1.5))
        mask = mask & ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                        ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) > (length * 1.3))
        mask = mask & (terrain.height_field_raw == int(0.0 / terrain.vertical_scale))
        terrain.height_field_raw[mask] = int(height / terrain.vertical_scale)
        if idx == 1:
            goals[-1] = [corner[1] / terrain.horizontal_scale + center_y, corner[0] / terrain.horizontal_scale + center_x]
            corner2 = (
                np.sign(corner[0]) * (np.abs(corner[0]) + 2.2), np.sign(corner[1]) * (np.abs(corner[1]) + 2.2))
            target_x2 = corner2[0] / terrain.horizontal_scale + center_x
            target_y2 = corner2[1] / terrain.horizontal_scale + center_y
            # mask = (((np.abs(x_indices - target_x2) * terrain.horizontal_scale) < 0.25) &
            #         ((np.abs(y_indices - target_y2) * terrain.horizontal_scale) < 0.25))
            # terrain.height_field_raw[mask] = int(0.2 / terrain.vertical_scale)
        elif idx < 3:
            corner2 = (
                np.sign(corner[0]) * (np.abs(corner[0]) + 2.2), np.sign(corner[1]) * (np.abs(corner[1]) + 2.2))
            target_x2 = corner2[0] / terrain.horizontal_scale + center_x
            target_y2 = corner2[1] / terrain.horizontal_scale + center_y
            # mask = (((np.abs(x_indices - target_x2) * terrain.horizontal_scale) < 0.25) &
            #         ((np.abs(y_indices - target_y2) * terrain.horizontal_scale) < 0.25))
            # terrain.height_field_raw[mask] = int(-0.5 / terrain.vertical_scale)
    terrain.goals = goals * terrain.horizontal_scale
    mask = (np.abs(((x_indices - center_x) * terrain.horizontal_scale)-
             ((y_indices - center_y) * terrain.horizontal_scale)) < 0.6)
    mask = mask | (np.abs(((x_indices - center_x) * terrain.horizontal_scale)+
             ((y_indices - center_y) * terrain.horizontal_scale)) < 0.6)
    mask = mask & (terrain.height_field_raw >= int(0.05 / terrain.vertical_scale))
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)
    terrain.stone_heights = np.array(heights_in_order)
    return heights_in_order


def parkour_middle_choose_terrain_paper(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           target_radius_coeff=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           length=1.8,
                           terrain_length=16.0,
                           flat=False,
                           max_difficulty=False):
    goals = np.zeros((num_stones + 2, 2))
    height, width = terrain.height_field_raw.shape
    center_x = width // 2
    center_y = height // 2
    terrain.height_field_raw[:] = int(pad_height // terrain.vertical_scale)
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    # 找到超出半径的区域
    # length = np.random.uniform(2.0, 3.0)
    # length = np.random.uniform(2.0, 2.4)
    # length = np.random.uniform(2.0, 2.2)
    # length = 1.8
    # length = np.random.uniform(2.9, 3.0)
    # length = np.random.uniform(2.5, 3.0)
    w = 0.4
    mask = ((np.abs(x_indices - center_x) * terrain.horizontal_scale) < length) & ((np.abs(y_indices - center_y) * terrain.horizontal_scale) < length)

    # 应用掩码设置高度
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)

    cord = length - w - 0.1
    corners = [(0, (cord, cord)), (1, (-cord, -cord)), (2, (cord, -cord)), (3, (-cord, cord))]
    np.random.shuffle(corners)
    # corners.append((3, (-cord, cord)))
    # difficulty_level = np.random.uniform(0, 1)
    height_0 = np.random.uniform(0.0, 0.5)  # 0.00 ~ 0.5           # 0.05
    height_1 = height_0 + np.random.uniform(0.1, 0.25)  # 0.1 ~ 0.75  # 0.15
    height_2 = height_1 + np.random.uniform(0.1, 0.15)  # 0.2 ~ 0.9   # 0.25
    # height_0 = np.random.uniform(0.05, 0.5)  # 0.05 ~ 0.5           # 0.05
    # height_1 = height_0 + np.random.uniform(0.1, 0.101 + 0.15 * difficulty_level)  # 0.1 ~ 0.75  # 0.15
    # height_2 = height_1 + np.random.uniform(0.1, 0.101 + 0.05 * difficulty_level)  # 0.2 ~ 0.9   # 0.25
    height_3 = 0.0
    if max_difficulty:
        height_0 = 0.25
        height_1 = 0.37
        height_2 = 0.49
        height_3 = 0.0
    # if np.random.rand() < 0.85:
    #     # height_0 = np.random.uniform(0.1, 0.2) # 0.1 ~ 0.2
    #     # height_1 = height_0 + np.random.uniform(0.1, 0.15) # 0.2 ~ 0.35
    #     # height_2 = height_1 + np.random.uniform(0.1, 0.15) # 0.3 ~ 0.5
    #     # height_3 = 0.0
    #
    # else:
    #     height_0 = np.random.uniform(-0.15, -0.05)
    #     height_1 = np.random.uniform(0.1, 0.2)
    #     height_2 = height_1 + np.random.uniform(0.1, 0.15)
    #     height_3 = 0.0
    heights = [height_0, height_1, height_2, height_3]
    heights_in_order = [(item0[0], item) for item0, item in zip(corners, heights)]
    heights_in_order = sorted(heights_in_order, key=lambda x: x[0])
    heights_in_order = [item[1] for item in heights_in_order]
    for idx, (corner, height) in enumerate(zip(corners, heights)):
        corner = corner[1]
        corner = (np.sign(corner[0]) * (np.abs(corner[0]) + w), np.sign(corner[1]) * (np.abs(corner[1]) + w))
        target_x = corner[0] / terrain.horizontal_scale + center_x
        target_y = corner[1] / terrain.horizontal_scale + center_y
        mask = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) +
                (np.abs(y_indices - target_y) * terrain.horizontal_scale) < 1.0)
        terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)

        mask = ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) < (length * 1.5))
        mask = mask & ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                        ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) > (length * 1.3))
        mask = mask & (terrain.height_field_raw == int(0.0 / terrain.vertical_scale))
        terrain.height_field_raw[mask] = int(height / terrain.vertical_scale)
        if idx == 1:
            goals[-1] = [corner[1] / terrain.horizontal_scale + center_y, corner[0] / terrain.horizontal_scale + center_x]
            corner2 = (
                np.sign(corner[0]) * (np.abs(corner[0]) + 2.2), np.sign(corner[1]) * (np.abs(corner[1]) + 2.2))
            target_x2 = corner2[0] / terrain.horizontal_scale + center_x
            target_y2 = corner2[1] / terrain.horizontal_scale + center_y
            # mask = (((np.abs(x_indices - target_x2) * terrain.horizontal_scale) < 0.25) &
            #         ((np.abs(y_indices - target_y2) * terrain.horizontal_scale) < 0.25))
            # terrain.height_field_raw[mask] = int(0.2 / terrain.vertical_scale)
        elif idx < 3:
            corner2 = (
                np.sign(corner[0]) * (np.abs(corner[0]) + 2.2), np.sign(corner[1]) * (np.abs(corner[1]) + 2.2))
            target_x2 = corner2[0] / terrain.horizontal_scale + center_x
            target_y2 = corner2[1] / terrain.horizontal_scale + center_y
            # mask = (((np.abs(x_indices - target_x2) * terrain.horizontal_scale) < 0.25) &
            #         ((np.abs(y_indices - target_y2) * terrain.horizontal_scale) < 0.25))
            # terrain.height_field_raw[mask] = int(-0.5 / terrain.vertical_scale)
    terrain.goals = goals * terrain.horizontal_scale
    mask = (np.abs(((x_indices - center_x) * terrain.horizontal_scale)-
             ((y_indices - center_y) * terrain.horizontal_scale)) < 0.6)
    mask = mask | (np.abs(((x_indices - center_x) * terrain.horizontal_scale)+
             ((y_indices - center_y) * terrain.horizontal_scale)) < 0.6)
    mask = mask & (terrain.height_field_raw >= int(0.05 / terrain.vertical_scale))
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)
    terrain.stone_heights = np.array(heights_in_order)

    mask_3 = terrain.height_field_raw > int(-0.6 / terrain.vertical_scale)
    dist_to_region = ndimage.distance_transform_edt(~mask_3)
    # 满足距离条件的位置设为 True，其余为 False
    mask_3 = dist_to_region >= 10
    sss = terrain.height_field_raw[mask_3]
    rnd = np.random.uniform(-0.05, 0.05, sss.shape)
    terrain.height_field_raw[mask_3] = np.round((rnd - 0.8) / terrain.vertical_scale)
    return heights_in_order

def parkour_middle_choose_terrain_0106(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           target_radius_coeff=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           terrain_length=16.0,
                           flat=False):
    goals = np.zeros((num_stones + 2, 2))
    height, width = terrain.height_field_raw.shape
    center_x = width // 2
    center_y = height // 2
    terrain.height_field_raw[:] = int(pad_height // terrain.vertical_scale)
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    # 找到超出半径的区域
    length = 3.
    w = 0.4
    mask = ((np.abs(x_indices - center_x) * terrain.horizontal_scale) < length) & ((np.abs(y_indices - center_y) * terrain.horizontal_scale) < length)

    # 应用掩码设置高度
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)

    cord = length - w - 0.1
    corners = [(cord, cord), (-cord, -cord), (cord, -cord), (-cord, cord)]
    np.random.shuffle(corners)
    if np.random.rand() < 0.85:
        height_0 = np.random.uniform(0.1, 0.2)
        height_1 = height_0 + np.random.uniform(0.1, 0.15)
        height_2 = height_1 + np.random.uniform(0.1, 0.15)
        height_3 = 0.0
    else:
        height_0 = np.random.uniform(-0.15, -0.05)
        height_1 = np.random.uniform(0.1, 0.2)
        height_2 = height_1 + np.random.uniform(0.1, 0.15)
        height_3 = 0.0
    heights = [height_0, height_1, height_2, height_3]
    for idx, (corner, height) in enumerate(zip(corners, heights)):
        corner = (
        np.sign(corner[0]) * (np.abs(corner[0]) + w), np.sign(corner[1]) * (np.abs(corner[1]) + w))
        target_x = corner[0] / terrain.horizontal_scale + center_x
        target_y = corner[1] / terrain.horizontal_scale + center_y
        mask = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) +
                (np.abs(y_indices - target_y) * terrain.horizontal_scale) < 1.0)
        terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)

        mask = ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) < (3 * 1.5))
        mask = mask & ((((x_indices - center_x) * terrain.horizontal_scale) * np.sign(corner[0]) +
                        ((y_indices - center_y) * terrain.horizontal_scale) * np.sign(corner[1])) > (3 * 1.35))
        mask = mask & (terrain.height_field_raw == int(0.0 / terrain.vertical_scale))
        terrain.height_field_raw[mask] = int(height / terrain.vertical_scale)
        if idx == 1:
            goals[-1] = [corner[1] / terrain.horizontal_scale + center_y, corner[0] / terrain.horizontal_scale + center_x]
            corner2 = (
                np.sign(corner[0]) * (np.abs(corner[0]) + 2.2), np.sign(corner[1]) * (np.abs(corner[1]) + 2.2))
            target_x2 = corner2[0] / terrain.horizontal_scale + center_x
            target_y2 = corner2[1] / terrain.horizontal_scale + center_y
            mask = (((np.abs(x_indices - target_x2) * terrain.horizontal_scale) < 0.25) &
                    ((np.abs(y_indices - target_y2) * terrain.horizontal_scale) < 0.25))
            terrain.height_field_raw[mask] = int(0.2 / terrain.vertical_scale)
        elif idx < 3:
            corner2 = (
                np.sign(corner[0]) * (np.abs(corner[0]) + 2.2), np.sign(corner[1]) * (np.abs(corner[1]) + 2.2))
            target_x2 = corner2[0] / terrain.horizontal_scale + center_x
            target_y2 = corner2[1] / terrain.horizontal_scale + center_y
            mask = (((np.abs(x_indices - target_x2) * terrain.horizontal_scale) < 0.25) &
                    ((np.abs(y_indices - target_y2) * terrain.horizontal_scale) < 0.25))
            terrain.height_field_raw[mask] = int(-0.5 / terrain.vertical_scale)
    terrain.goals = goals * terrain.horizontal_scale
    mask = (np.abs(((x_indices - center_x) * terrain.horizontal_scale)-
             ((y_indices - center_y) * terrain.horizontal_scale)) < 0.5)
    mask = mask | (np.abs(((x_indices - center_x) * terrain.horizontal_scale)+
             ((y_indices - center_y) * terrain.horizontal_scale)) < 0.5)
    mask = mask & (terrain.height_field_raw >= int(0.05 / terrain.vertical_scale))
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)

def parkour_middle_choose_terrain_old(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           target_radius_coeff=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           terrain_length=16.0,
                           flat=False):
    goals = np.zeros((num_stones + 2, 2))
    height, width = terrain.height_field_raw.shape
    center_x = width // 2
    center_y = height // 2
    terrain.height_field_raw[:] = int(pad_height // terrain.vertical_scale)
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    # 找到超出半径的区域
    length = 3.
    w = 0.4
    mask = ((np.abs(x_indices - center_x) * terrain.horizontal_scale) < length) & ((np.abs(y_indices - center_y) * terrain.horizontal_scale) < length)

    # 应用掩码设置高度
    terrain.height_field_raw[mask] = int(0.0 / terrain.vertical_scale)

    cord = length - w - 0.2
    corners = [(cord, cord), (-cord, -cord), (cord, -cord), (-cord, cord)]
    np.random.shuffle(corners)

    height_0 = np.random.uniform(0.15, 0.35)
    height_1 = height_0 + np.random.uniform(0.1, 0.2)
    height_2 = height_1 + np.random.uniform(0.1, 0.15)
    height_3 = 0.0
    heights = [height_0, height_1, height_2, height_3]
    boarder_3 = w
    # boarder_4 = 0.24
    for idx, (corner, height) in enumerate(zip(corners[:3], heights[:3])):
        target_x = corner[0] / terrain.horizontal_scale + center_x
        target_y = corner[1] / terrain.horizontal_scale + center_y
        mask = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) < boarder_3) & (
                (np.abs(y_indices - target_y) * terrain.horizontal_scale) < boarder_3)
        terrain.height_field_raw[mask] = int(height / terrain.vertical_scale)
        # mask = ((np.abs(x_indices - target_x) * terrain.horizontal_scale) < boarder_4) & (
        #         (np.abs(y_indices - target_y) * terrain.horizontal_scale) < boarder_4)
        # if not idx == 1:
        #     terrain.height_field_raw[mask] = int(-0.8 / terrain.vertical_scale)

    goals[-1] = [corners[1][1] / terrain.horizontal_scale + center_y, corners[1][0] / terrain.horizontal_scale + center_x]
    terrain.goals = goals * terrain.horizontal_scale


def parkour_vertical_line_plat_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           target_radius_coeff=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           terrain_length=16.0,
                           flat=False):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200

    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height
    pad_radius = (terrain_length / 2) * 0.82 # 6.5

    target_radius = np.random.uniform(0.89, 0.9) * pad_radius
    target_angle = np.random.randint(0, 2) * np.pi

    target_x = target_radius * np.cos(target_angle)
    target_y = target_radius * np.sin(target_angle)

    goal_x = (target_radius - 0.5) * np.cos(target_angle)
    goal_y = (target_radius - 0.5) * np.sin(target_angle)

    # stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)

    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len + int(terrain_length / 2 / terrain.horizontal_scale)
    goals[0] = [dis_x - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 2.0 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    target_x = round(target_x / terrain.horizontal_scale)
    target_y = round(target_y / terrain.horizontal_scale)
    half_bound = round(0.1 / terrain.horizontal_scale)

    # # 获取地形的宽和高
    height, width = terrain.height_field_raw.shape

    # 中心点
    center_x = width // 2
    center_y = height // 2


    goal_x = round(goal_x / terrain.horizontal_scale)
    goal_y = round(goal_y / terrain.horizontal_scale)

    goal_x = center_y + goal_x
    goal_y = center_x + goal_y

    goals[-1] = [goal_x, goal_y]
    terrain.goals = goals * terrain.horizontal_scale

    pad_height = int(pad_height // terrain.vertical_scale)
    #
    # 创建坐标网格
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))

    # 计算到中心的距离
    distances = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2) * terrain.horizontal_scale

    # 找到超出半径的区域
    mask = ((np.abs(x_indices - center_x) * terrain.horizontal_scale) > 0.8)

    # 应用掩码设置高度
    terrain.height_field_raw[mask] = pad_height
    pad_height2 = int(1.0 // terrain.vertical_scale)

    pad_width = int(pad_width // terrain.horizontal_scale)
    # pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[(pad_width//2):pad_width, :] = pad_height2
    terrain.height_field_raw[-pad_width:(-pad_width//2), :] = pad_height2
    terrain.height_field_raw[:pad_width//2, :] = pad_height
    terrain.height_field_raw[-pad_width//2:, :] = pad_height
    terrain.height_field_raw[center_y + target_x - half_bound:center_y +target_x + half_bound, center_x + target_y - half_bound:center_x +  target_y + half_bound] = int(2.5 // terrain.vertical_scale)



def parkour_S_line_terrain(terrain,
                           platform_len=2.5,
                           platform_height=0.,
                           num_stones=8,
                           target_radius_coeff=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           terrain_length=16.0,
                           flat=False):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200

    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(-0.8 / terrain.vertical_scale)
    terrain.height_field_raw[:] = platform_height
    pad_radius = (terrain_length / 2) * 0.82 # 6.5

    # horizon_line_length = 0.8 * pad_radius
    line_length = 0.65 * pad_radius * 0.7
    # vertical_line_length = 0.65 * pad_radius * 0.7 * 1.3
    vertical_line_length = line_length
    # right side x: - 3.7
    # top side: y: -3.151
    # bottom size: y: 3.15

    # left side x: 4.0

    target_x = (np.random.randint(0, 2) * 2 - 1) * 2.4   # line_length * (np.random.randint(0, 2) * 2 - 1) * 0.85
    target_y = (np.random.randint(0, 2) * 2 - 1) * 1.55   # line_length * (np.random.randint(0, 2) * 2 - 1) * 0.85
    # if target_x > 0:
    #     target_x = 4.0
    goal_x = target_x
    goal_y = target_y


    target_x = round(target_x / terrain.horizontal_scale)
    target_y = round(target_y / terrain.horizontal_scale)
    half_bound = round(0.1 / terrain.horizontal_scale)

    # # 获取地形的宽和高
    height, width = terrain.height_field_raw.shape

    # 中心点
    center_x = width // 2
    center_y = height // 2


    goal_x = round(goal_x / terrain.horizontal_scale)
    goal_y = round(goal_y / terrain.horizontal_scale)

    goal_x = center_y + goal_x
    goal_y = center_x + goal_y

    goals[-1] = [goal_x, goal_y]
    terrain.goals = goals * terrain.horizontal_scale

    pad_height = int(pad_height // terrain.vertical_scale)
    #
    # 创建坐标网格
    row_indices, col_indices = np.meshgrid(np.arange(width), np.arange(height))

    road_width = 0.65
    mask = ((np.abs(col_indices - center_x) * terrain.horizontal_scale < road_width) &
            ((row_indices - center_y) * terrain.horizontal_scale > -road_width) &
            ((row_indices - center_y) * terrain.horizontal_scale < line_length))
    mask = mask | ((np.abs(row_indices - center_x) * terrain.horizontal_scale < road_width) & (
                (col_indices - center_x) * terrain.horizontal_scale > -road_width) & (
                               (col_indices - center_x) * terrain.horizontal_scale < line_length))
    mask = mask | ((np.abs(row_indices - center_x) * terrain.horizontal_scale < road_width) & (
                (col_indices - center_x) * terrain.horizontal_scale < road_width) & (
                               (col_indices - center_x) * terrain.horizontal_scale > -line_length))
    mask = mask | (
            (np.abs(col_indices - (
                        center_x + line_length / terrain.horizontal_scale)) * terrain.horizontal_scale < road_width) &
            ((row_indices - center_y) * terrain.horizontal_scale < road_width) &
            ((row_indices - center_y) * terrain.horizontal_scale > -vertical_line_length)
    )
    mask = mask | (
            (np.abs(col_indices - (
                    center_x + line_length / terrain.horizontal_scale)) * terrain.horizontal_scale < road_width) &
            ((row_indices - center_y) * terrain.horizontal_scale > -road_width) &
            ((row_indices - center_y) * terrain.horizontal_scale < vertical_line_length)
    )
    mask = mask | (
            (np.abs(col_indices - (
                    center_x - line_length / terrain.horizontal_scale)) * terrain.horizontal_scale < road_width) &
            ((row_indices - center_y) * terrain.horizontal_scale < road_width) &
            ((row_indices - center_y) * terrain.horizontal_scale > -vertical_line_length)
    )
    mask = mask | (
            (np.abs(col_indices - (
                    center_x - line_length / terrain.horizontal_scale)) * terrain.horizontal_scale < road_width) &
            ((row_indices - center_y) * terrain.horizontal_scale > -road_width) &
            ((row_indices - center_y) * terrain.horizontal_scale < vertical_line_length)
    )
    if target_x > 0:
        mask = mask | ((np.abs(col_indices - center_x) * terrain.horizontal_scale < road_width) &
                ((row_indices - center_y) * terrain.horizontal_scale > -2.5 * road_width) &
                ((row_indices - center_y) * terrain.horizontal_scale < line_length))
        # mask = mask | ((np.abs(row_indices - center_x) * terrain.horizontal_scale < road_width) & ((col_indices - center_x) * terrain.horizontal_scale > -road_width) & ((col_indices - center_x) * terrain.horizontal_scale < line_length))
        if target_y < 0:
            pass
            # mask = mask | (
            #     (np.abs(col_indices - (center_x + line_length / terrain.horizontal_scale)) * terrain.horizontal_scale < road_width) &
            #     ((row_indices - center_y) * terrain.horizontal_scale < road_width) &
            #     ((row_indices - center_y) * terrain.horizontal_scale > -line_length)
            # )
        else:
            mask = mask | ((np.abs(row_indices - center_x) * terrain.horizontal_scale < road_width) & (
                        (col_indices - center_x) * terrain.horizontal_scale > -road_width) & (
                                       (col_indices - center_x) * terrain.horizontal_scale < (line_length + 2.5 * road_width)))
            pass
            # mask = mask | (
            #         (np.abs(col_indices - (
            #                     center_x + line_length / terrain.horizontal_scale)) * terrain.horizontal_scale < road_width) &
            #         ((row_indices - center_y) * terrain.horizontal_scale > -road_width) &
            #         ((row_indices - center_y) * terrain.horizontal_scale < line_length)
            # )
    else:
        # mask = mask | ((np.abs(row_indices - center_x) * terrain.horizontal_scale < road_width) & ((col_indices - center_x) * terrain.horizontal_scale < road_width) & ((col_indices - center_x) * terrain.horizontal_scale > -line_length))
        if target_y < 0:
            pass
            # mask = mask | (
            #         (np.abs(col_indices - (
            #                     center_x - line_length / terrain.horizontal_scale)) * terrain.horizontal_scale < road_width) &
            #         ((row_indices - center_y) * terrain.horizontal_scale < road_width) &
            #         ((row_indices - center_y) * terrain.horizontal_scale > -line_length)
            # )
        else:
            mask = mask | ((np.abs(row_indices - center_x) * terrain.horizontal_scale < road_width) & (
                        (col_indices - center_x) * terrain.horizontal_scale < road_width) & (
                                       (col_indices - center_x) * terrain.horizontal_scale > (-line_length - 2 * road_width)))
            # mask = mask | (
            #         (np.abs(col_indices - (
            #                 center_x - line_length / terrain.horizontal_scale)) * terrain.horizontal_scale < road_width) &
            #         ((row_indices - center_y) * terrain.horizontal_scale > -road_width) &
            #         ((row_indices - center_y) * terrain.horizontal_scale < line_length)
            # )
    # 应用掩码设置高度
    terrain.height_field_raw[mask] = round(0.1 / terrain.vertical_scale)
    distance = distance_transform_edt(~mask)
    # 选择距离不超过N且原值为-0.8的点
    selected = (distance <= road_width / 4 / terrain.horizontal_scale) & (~mask)
    # 将这些点的值设为0.4
    terrain.height_field_raw[selected] = round(0.45 / terrain.vertical_scale)
    # half_bound = 0.1
    # terrain.height_field_raw[center_y + target_x - half_bound:center_y +target_x + half_bound, center_x + target_y - half_bound:center_x +  target_y + half_bound] = int(1.2 // terrain.vertical_scale)

    # pad_height2 = int(1.0 // terrain.vertical_scale)
    #
    # pad_width = int(pad_width // terrain.horizontal_scale)
    # # pad_height = int(pad_height // terrain.vertical_scale)
    # terrain.height_field_raw[:, :pad_width] = pad_height
    # terrain.height_field_raw[:, -pad_width:] = pad_height
    # terrain.height_field_raw[(pad_width//2):pad_width, :] = pad_height2
    # terrain.height_field_raw[-pad_width:(-pad_width//2), :] = pad_height2
    # terrain.height_field_raw[:pad_width//2, :] = pad_height
    # terrain.height_field_raw[-pad_width//2:, :] = pad_height
    # terrain.height_field_raw[center_y + target_x - half_bound:center_y +target_x + half_bound, center_x + target_y - half_bound:center_x +  target_y + half_bound] = int(2.5 // terrain.vertical_scale)





def parkour_step_terrain(terrain,
                           platform_len=2.5, 
                           platform_height=0., 
                           num_stones=8,
                        #    x_range=[1.5, 2.4],
                            x_range=[0.2, 0.4],
                           y_range=[-0.15, 0.15],
                           half_valid_width=[0.45, 0.5],
                           step_height = 0.2,
                           pad_width=0.1,
                           pad_height=0.5):
    goals = np.zeros((num_stones+2, 2))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round( (x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round( (x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    # stone_width = round(stone_width / terrain.horizontal_scale)
    
    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
        terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
        
        last_dis_x = dis_x
        goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2
    
    # hurdle
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height
    
    # step up
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    
    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    
    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    terrain.height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / terrain.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    terrain.height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    
    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int*terrain.horizontal_scale, 0., max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def half_platform_terrain(terrain, start2center=2, max_height=1):
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    terrain.height_field_raw[:, :] = max_height_int
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-1):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    def get_rand_dis_int(scale):
        return np.random.randint(int(- scale / terrain.horizontal_scale + 1), int(scale / terrain.horizontal_scale))
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0