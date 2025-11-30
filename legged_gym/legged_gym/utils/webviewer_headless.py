from typing import List, Optional

import logging
import math
import threading

import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

try:
    import flask
except ImportError:
    flask = None

try:
    import imageio
    import isaacgym
    import isaacgym.torch_utils as torch_utils
    from isaacgym import gymapi
except ImportError:
    imageio = None
    isaacgym = None
    torch_utils = None
    gymapi = None

try:
    import cv2

except ImportError:
    cv2 = None
from concurrent.futures import ThreadPoolExecutor


def overlay_depth_image(image, image_depth, scale_factor=2):
    """
    将 depth_image 等比例放大并覆盖到 image 的右上角。

    参数:
    - image: np.ndarray, 主图像 (H, W, 3)。
    - image_depth: np.ndarray, 深度图像 (h, w)，单通道。

    返回:
    - 无返回值，直接在 image 上修改。
    """
    # 确保输入图像形状符合要求
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入的 image 必须是 (H, W, 3) 的彩色图像！")
    if len(image_depth.shape) != 2:
        raise ValueError("输入的 image_depth 必须是单通道灰度图！")

    # 步骤 1: 将 depth_image 等比例放大 2 倍
    if scale_factor == 1:
        depth_resized = image_depth
    else:
        depth_resized = cv2.resize(image_depth, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # 步骤 2: 将放大的 depth_image 粘贴到 image 的右上角
    resized_height, resized_width = depth_resized.shape
    image_height, image_width, _ = image.shape

    # 计算右上角的起始点
    start_x = image_width - resized_width - int(1/16 * image_width)
    start_y = int(1/8 * image_height)

    # 确保覆盖的区域没有越界
    if start_x < 0 or start_y + resized_height > image_height:
        raise ValueError("放大的 depth_image 尺寸超出了 image 的范围！")

    # 如果 depth_image 是单通道，需要转为 3 通道
    depth_colored = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)

    # 将 depth_resized 粘贴到 image 的右上角
    image[start_y:start_y + resized_height, start_x:start_x + resized_width] = depth_colored


def add_text_to_image(image, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                      color=(255, 255, 255), thickness=2):
    """
    在图像的指定位置添加文字。

    参数:
    - image: np.ndarray, 输入图像 (H, W, 3)。
    - text: str, 要添加的文字。
    - position: tuple, 文字的左下角坐标 (x, y)。
    - font: OpenCV 字体类型，默认 cv2.FONT_HERSHEY_SIMPLEX。
    - font_scale: float, 字体大小缩放比例，默认 1。
    - color: tuple, 文字颜色 (B, G, R)，默认白色。
    - thickness: int, 文字线条厚度，默认 2。

    返回:
    - 无返回值，直接修改输入图像。
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入的图像必须是 (H, W, 3) 的彩色图像！")

    # 在图像上添加文字
    # cv2.putText(image, text, position, font, font_scale, color, thickness)
    umat_image = cv2.UMat(image)

    # 在图像上添加文字
    cv2.putText(umat_image, text, position, font, font_scale, color, thickness)

    # 将 cv2.UMat 转回 np.ndarray 并返回
    return umat_image.get()

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r) if r != 0 else 0
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

class WebViewer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555) -> None:
        """
        Web viewer for Isaac Gym

        :param host: Host address (default: "127.0.0.1")
        :type host: str
        :param port: Port number (default: 5000)
        :type port: int
        """

        self._log = logging.getLogger('werkzeug')
        self._log.disabled = True

        self._image = None
        self._image_depth = None
        self._camera_id = 0
        self._stopped = False
        self._camera_type = gymapi.IMAGE_COLOR
        # self._camera_type = gymapi.IMAGE_DEPTH
        self._notified = False
        self._wait_for_page = True
        self._pause_stream = False
        self._images_list = []
        self._event_load = threading.Event()
        self._event_stream = threading.Event()
        self._event_stream_depth = threading.Event()


    def attach_view_camera(self, i, env_handle, actor_handle, root_pos):
        if True:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 960
            camera_props.height = 540
            # camera_props.enable_tensors = True
            # camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self._gym.create_camera_sensor(env_handle, camera_props)
            self._cameras.append(camera_handle)
            
            cam_pos = root_pos + np.array([0, 1, 0.5])
            self._gym.set_camera_location(camera_handle, env_handle, gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))

    def setup(self, env) -> None:
        """Setup the web viewer

        :param gym: The gym
        :type gym: isaacgym.gymapi.Gym
        :param sim: Simulation handle
        :type sim: isaacgym.gymapi.Sim
        :param envs: Environment handles
        :type envs: list of ints
        :param cameras: Camera handles
        :type cameras: list of ints
        """
        self._gym = env.gym
        self._sim = env.sim
        self._envs = env.envs
        self._cameras = []
        self._env = env
        self.cam_pos_rel = np.array([0, 2, 1]) * 2.0
        for i in range(self._env.num_envs):
            root_pos = self._env.root_states[i, :3].cpu().numpy()
            self.attach_view_camera(i, self._envs[i], self._env.actor_handles[i], root_pos)
    
    def render(self,
               fetch_results: bool = True,
               step_graphics: bool = True,
               render_all_camera_sensors: bool = True,
               wait_for_page_load: bool = True, **kwargs) -> None:
        """Render and get the image from the current camera

        This function must be called after the simulation is stepped (post_physics_step).
        The following Isaac Gym functions are called before get the image.
        Their calling can be skipped by setting the corresponding argument to False

        - fetch_results
        - step_graphics
        - render_all_camera_sensors

        :param fetch_results: Call Gym.fetch_results method (default: True)
        :type fetch_results: bool
        :param step_graphics: Call Gym.step_graphics method (default: True)
        :type step_graphics: bool
        :param render_all_camera_sensors: Call Gym.render_all_camera_sensors method (default: True)
        :type render_all_camera_sensors: bool
        :param wait_for_page_load: Wait for the page to load (default: True)
        :type wait_for_page_load: bool
        """


        # isaac gym API
        if fetch_results:
            self._gym.fetch_results(self._sim, True)
        if step_graphics:
            self._gym.step_graphics(self._sim)
        if render_all_camera_sensors:
            self._gym.render_all_camera_sensors(self._sim)

        # get image
        image = self._gym.get_camera_image(self._sim,
                                           self._envs[self._camera_id],
                                           self._cameras[self._camera_id],
                                           self._camera_type)
        if self._camera_type == gymapi.IMAGE_COLOR:
            self._image = image.reshape(image.shape[0], -1, 4)[..., :3]
        elif self._camera_type == gymapi.IMAGE_DEPTH:
            self._image = -image.reshape(image.shape[0], -1)
            minimum = 0 if np.isinf(np.min(self._image)) else np.min(self._image)
            maximum = 5 if np.isinf(np.max(self._image)) else np.max(self._image)
            self._image = np.clip(1 - (self._image - minimum) / (maximum - minimum), 0, 1)
            self._image = np.uint8(255 * self._image)
        else:
            raise ValueError("Unsupported camera type")
        # self._image = np.uint8(self._image)
        if self._env.cfg.depth.use_camera:
            self._image_depth = (self._env.depth_buffer[self._camera_id, -1].cpu().numpy() + 1) / 2
            self._image_depth = np.uint8(255 * self._image_depth)
            overlay_depth_image(self._image, self._image_depth)
        # print(f'{isinstance(self._image, np.ndarray)}, {len(self._image.shape)}')
        # self._image = cv2.UMat(self._image)
        # print(f'image type: {type(self._image)}, image dtype: {self._image.dtype}')
        t = self._env.dt * int(self._env.episode_length_buf[self._camera_id])
        text = f'step: {t: .2f}'
        if 'target_std' in kwargs:
            text += f', uncertainty: {kwargs["target_std"]:.3f}, ebd_diff: {kwargs["ebd_diff"]:.3f}'

        self._image = add_text_to_image(self._image, text)
        
        root_pos = self._env.root_states[self._camera_id, :3].cpu().numpy()
        cam_pos = root_pos + self.cam_pos_rel
        self._gym.set_camera_location(self._cameras[self._camera_id], self._envs[self._camera_id], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))

    def render_all(self,
               fetch_results: bool = True,
               step_graphics: bool = True,
               render_all_camera_sensors: bool = True,
               wait_for_page_load: bool = True, **kwargs) -> None:
        """Render and get the image from the current camera

        This function must be called after the simulation is stepped (post_physics_step).
        The following Isaac Gym functions are called before get the image.
        Their calling can be skipped by setting the corresponding argument to False

        - fetch_results
        - step_graphics
        - render_all_camera_sensors

        :param fetch_results: Call Gym.fetch_results method (default: True)
        :type fetch_results: bool
        :param step_graphics: Call Gym.step_graphics method (default: True)
        :type step_graphics: bool
        :param render_all_camera_sensors: Call Gym.render_all_camera_sensors method (default: True)
        :type render_all_camera_sensors: bool
        :param wait_for_page_load: Wait for the page to load (default: True)
        :type wait_for_page_load: bool
        """
        dt = self._env.dt

        # isaac gym API
        if fetch_results:
            self._gym.fetch_results(self._sim, True)
        if step_graphics:
            self._gym.step_graphics(self._sim)
        if render_all_camera_sensors:
            self._gym.render_all_camera_sensors(self._sim)

        # get image
        image_list = [None for _ in range(self._env.num_envs)]
        images_depth_list = [None for _ in range(self._env.num_envs)]
        def modify_image_i(env_id):
            image = self._gym.get_camera_image(self._sim,
                                               self._envs[env_id],
                                               self._cameras[env_id],
                                               self._camera_type)
            if self._camera_type == gymapi.IMAGE_COLOR:
                image = image.reshape(image.shape[0], -1, 4)[..., :3]
            elif self._camera_type == gymapi.IMAGE_DEPTH:
                image = -image.reshape(image.shape[0], -1)
                minimum = 0 if np.isinf(np.min(image)) else np.min(image)
                maximum = 5 if np.isinf(np.max(image)) else np.max(image)
                image = np.clip(1 - (image - minimum) / (maximum - minimum), 0, 1)
                image = np.uint8(255 * image)
            else:
                raise ValueError("Unsupported camera type")
            # self._image = np.uint8(self._image)
            if self._env.cfg.depth.use_camera:
                image_depth = (self._env.depth_buffer[env_id, -1].cpu().numpy() + 1) / 2
                image_depth = np.uint8(255 * image_depth)
                overlay_depth_image(image, image_depth)
                images_depth_list[env_id] = image_depth
            else:
                scan_dots = torch.clamp((self._env.measured_heights[env_id] - (-0.8)) / (1.5 - (-0.8)), 0, 1)
                scan_dots = scan_dots.reshape((12, 11))
                scan_dots = np.uint8(255 * scan_dots.cpu().numpy())
                scan_dots = scan_dots[::-1]
                scan_dots = scan_dots[:, ::-1]
                overlay_depth_image(image, scan_dots, 10)
                # images_depth_list[env_id] = image_depth

            # print(f'{isinstance(self._image, np.ndarray)}, {len(self._image.shape)}')
            # self._image = cv2.UMat(self._image)
            # print(f'image type: {type(self._image)}, image dtype: {self._image.dtype}')
            t = dt * int(self._env.episode_length_buf[env_id])
            text = f'step: {t: .2f}'
            if 'target_std' in kwargs:
                text += f', uncertainty: {kwargs["target_std"][env_id].item():.3f}, ebd_diff: {kwargs["ebd_diff"][env_id].item():.3f}'

            image = add_text_to_image(image, text)
            text_2 = None
            if 'ep_return' in kwargs:
                text_2 = f'ep_return: {float(kwargs["ep_return"][env_id].item()):.3f}'
            if text_2 is not None:
                image = add_text_to_image(image, text_2, (10, 75))
            root_pos = self._env.root_states[env_id, :3].cpu().numpy()
            cam_pos = root_pos + self.cam_pos_rel
            self._gym.set_camera_location(self._cameras[env_id], self._envs[env_id],
                                          gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
            image_list[env_id] = image
            return env_id
        with ThreadPoolExecutor(max_workers=15) as executor:
            results = list(executor.map(modify_image_i, range(self._env.num_envs)))
        self._images_list = image_list
        self._image = self._images_list[self._camera_id]
        if self._env.cfg.depth.use_camera:
            self._image_depth = images_depth_list[self._camera_id]


def ik(jacobian_end_effector: torch.Tensor,
       current_position: torch.Tensor,
       current_orientation: torch.Tensor,
       goal_position: torch.Tensor,
       goal_orientation: Optional[torch.Tensor] = None,
       damping_factor: float = 0.05,
       squeeze_output: bool = True) -> torch.Tensor:
    """
    Inverse kinematics using damped least squares method

    :param jacobian_end_effector: End effector's jacobian
    :type jacobian_end_effector: torch.Tensor
    :param current_position: End effector's current position
    :type current_position: torch.Tensor
    :param current_orientation: End effector's current orientation
    :type current_orientation: torch.Tensor
    :param goal_position: End effector's goal position
    :type goal_position: torch.Tensor
    :param goal_orientation: End effector's goal orientation (default: None)
    :type goal_orientation: torch.Tensor or None
    :param damping_factor: Damping factor (default: 0.05)
    :type damping_factor: float
    :param squeeze_output: Squeeze output (default: True)
    :type squeeze_output: bool

    :return: Change in joint angles
    :rtype: torch.Tensor
    """
    if goal_orientation is None:
        goal_orientation = current_orientation

    # compute error
    q = torch_utils.quat_mul(goal_orientation, torch_utils.quat_conjugate(current_orientation))
    error = torch.cat([goal_position - current_position,  # position error
                       q[:, 0:3] * torch.sign(q[:, 3]).unsqueeze(-1)],  # orientation error
                      dim=-1).unsqueeze(-1)

    # solve damped least squares (dO = J.T * V)
    transpose = torch.transpose(jacobian_end_effector, 1, 2)
    lmbda = torch.eye(6, device=jacobian_end_effector.device) * (damping_factor ** 2)
    if squeeze_output:
        return (transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error).squeeze(dim=2)
    else:
        return transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error

def print_arguments(args):
    print("")
    print("Arguments")
    for a in args.__dict__:
        print(f"  |-- {a}: {args.__getattribute__(a)}")

def print_asset_options(asset_options: 'isaacgym.gymapi.AssetOptions', asset_name: str = ""):
    attrs = ["angular_damping", "armature", "collapse_fixed_joints", "convex_decomposition_from_submeshes",
             "default_dof_drive_mode", "density", "disable_gravity", "fix_base_link", "flip_visual_attachments",
             "linear_damping", "max_angular_velocity", "max_linear_velocity", "mesh_normal_mode", "min_particle_mass",
             "override_com", "override_inertia", "replace_cylinder_with_capsule", "tendon_limit_stiffness", "thickness",
             "use_mesh_materials", "use_physx_armature", "vhacd_enabled"]  # vhacd_params
    print("\nAsset options{}".format(f" ({asset_name})" if asset_name else ""))
    for attr in attrs:
        print("  |-- {}: {}".format(attr, getattr(asset_options, attr) if hasattr(asset_options, attr) else "--"))
        # vhacd attributes
        if attr == "vhacd_enabled" and hasattr(asset_options, attr) and getattr(asset_options, attr):
            vhacd_attrs = ["alpha", "beta", "concavity", "convex_hull_approximation", "convex_hull_downsampling",
                           "max_convex_hulls", "max_num_vertices_per_ch", "min_volume_per_ch", "mode", "ocl_acceleration",
                           "pca", "plane_downsampling", "project_hull_vertices", "resolution"]
            print("  |-- vhacd_params:")
            for vhacd_attr in vhacd_attrs:
                print("  |   |-- {}: {}".format(vhacd_attr, getattr(asset_options.vhacd_params, vhacd_attr) \
                    if hasattr(asset_options.vhacd_params, vhacd_attr) else "--"))

def print_sim_components(gym, sim):
    print("")
    print("Sim components")
    print("  |--  env count:", gym.get_env_count(sim))
    print("  |--  actor count:", gym.get_sim_actor_count(sim))
    print("  |--  rigid body count:", gym.get_sim_rigid_body_count(sim))
    print("  |--  joint count:", gym.get_sim_joint_count(sim))
    print("  |--  dof count:", gym.get_sim_dof_count(sim))
    print("  |--  force sensor count:", gym.get_sim_force_sensor_count(sim))

def print_env_components(gym, env):
    print("")
    print("Env components")
    print("  |--  actor count:", gym.get_actor_count(env))
    print("  |--  rigid body count:", gym.get_env_rigid_body_count(env))
    print("  |--  joint count:", gym.get_env_joint_count(env))
    print("  |--  dof count:", gym.get_env_dof_count(env))

def print_actor_components(gym, env, actor):
    print("")
    print("Actor components")
    print("  |--  rigid body count:", gym.get_actor_rigid_body_count(env, actor))
    print("  |--  joint count:", gym.get_actor_joint_count(env, actor))
    print("  |--  dof count:", gym.get_actor_dof_count(env, actor))
    print("  |--  actuator count:", gym.get_actor_actuator_count(env, actor))
    print("  |--  rigid shape count:", gym.get_actor_rigid_shape_count(env, actor))
    print("  |--  soft body count:", gym.get_actor_soft_body_count(env, actor))
    print("  |--  tendon count:", gym.get_actor_tendon_count(env, actor))

def print_dof_properties(gymapi, props):
    print("")
    print("DOF properties")
    print("  |--  hasLimits:", props["hasLimits"])
    print("  |--  lower:", props["lower"])
    print("  |--  upper:", props["upper"])
    print("  |--  driveMode:", props["driveMode"])
    print("  |      |-- {}: gymapi.DOF_MODE_NONE".format(int(gymapi.DOF_MODE_NONE)))
    print("  |      |-- {}: gymapi.DOF_MODE_POS".format(int(gymapi.DOF_MODE_POS)))
    print("  |      |-- {}: gymapi.DOF_MODE_VEL".format(int(gymapi.DOF_MODE_VEL)))
    print("  |      |-- {}: gymapi.DOF_MODE_EFFORT".format(int(gymapi.DOF_MODE_EFFORT)))
    print("  |--  stiffness:", props["stiffness"])
    print("  |--  damping:", props["damping"])
    print("  |--  velocity (max):", props["velocity"])
    print("  |--  effort (max):", props["effort"])
    print("  |--  friction:", props["friction"])
    print("  |--  armature:", props["armature"])

def print_links_and_dofs(gym, asset):
    link_dict = gym.get_asset_rigid_body_dict(asset)
    dof_dict = gym.get_asset_dof_dict(asset)

    print("")
    print("Links")
    for k in link_dict:
        print(f"  |-- {k}: {link_dict[k]}")
    print("DOFs")
    for k in dof_dict:
        print(f"  |-- {k}: {dof_dict[k]}")
