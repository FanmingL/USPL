import shutil
from typing import List, Optional

import logging
import math
import threading

import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import pickle
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
import matplotlib
matplotlib.use('Agg')
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.patches import Polygon
from PIL import Image


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
    start_x = image_width - resized_width - int(1/20 * image_width)
    start_y = int(1/16 * image_height)

    # 确保覆盖的区域没有越界
    if start_x < 0 or start_y + resized_height > image_height:
        raise ValueError("放大的 depth_image 尺寸超出了 image 的范围！")

    # 如果 depth_image 是单通道，需要转为 3 通道
    depth_colored = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)

    # 将 depth_resized 粘贴到 image 的右上角
    image[start_y:start_y + resized_height, start_x:start_x + resized_width] = depth_colored


def add_text_to_image(image, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                      color=(255, 255, 255), thickness=2, force_add=False):
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
    if not force_add:
        return image
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
    def __init__(self, output_dir, maximum_camera_num, view_mode='square', include_depth=False, host: str = "127.0.0.1", port: int = 5555) -> None:
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
        self._include_depth = include_depth
        self._camera_id = 0
        self._stopped = False
        self._camera_type = gymapi.IMAGE_COLOR
        # self._camera_type = gymapi.IMAGE_DEPTH
        self._notified = False
        self._wait_for_page = True
        self._pause_stream = False
        self._view_mode = view_mode
        self._images_list = []
        self._event_load = threading.Event()
        self._event_stream = threading.Event()
        self._event_stream_depth = threading.Event()
        self.output_dir = output_dir
        self.processing_dir = os.path.join(output_dir, "processing_video")
        self.complete_dir = os.path.join(output_dir, "complete_video")
        self.attachment_dir = os.path.join(output_dir, "attachments")
        os.makedirs(self.processing_dir, exist_ok=True)
        os.makedirs(self.complete_dir, exist_ok=True)
        os.makedirs(self.attachment_dir, exist_ok=True)
        self.resize_ratio = 2.0
        self.maximum_camera_num = maximum_camera_num



    def attach_view_camera(self, i, env_handle, actor_handle, root_pos):
        if True:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 960 * 2
            camera_props.height = 540 * 2
            # camera_props.enable_tensors = True
            camera_props.horizontal_fov = 120

            camera_handle = self._gym.create_camera_sensor(env_handle, camera_props)
            self._cameras[i] = camera_handle
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

        self._env = env
        self._cameras = [None for _ in range(self._env.num_envs)]
        self.cam_pos_rel = np.array([0, 2, 1]) * 2.5

        rng = np.random.RandomState(seed=42)
        self._focused_camera_ids = rng.permutation(self._env.num_envs)[:self.maximum_camera_num]
        self._focused_camera_ids = sorted(self._focused_camera_ids)
        for i in self._focused_camera_ids:
            root_pos = self._env.root_states[i, :3].cpu().numpy()
            self.attach_view_camera(i, self._envs[i], self._env.actor_handles[i], root_pos)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writers = [None for _ in range(self._env.num_envs)]
        self._attachment_buffers = [None for _ in range(self._env.num_envs)]
        self.fig_lock = threading.Lock()

    @staticmethod
    def overlay_matplotlib_image(img_rgb, x_offset=10, y_offset=10,
                                 mean_x=0, mean_y=0,
                                 std_x=0, std_y=0, ego_x=0, ego_y=0,
                                 ego_yaw=0, tgt_x=0, tgt_y=0, xlim=None, ylim=None, xlim_axis=None, ylim_axis=None, fig_lock=None):
        """
        在输入的 img_rgb 图像上叠加一个 matplotlib 生成的图像（例如二维高斯分布的等概率线图）。
        还在图像中添加了ego的位置和朝向，以及tgt的位置（五角星标记）。

        参数:
            img_rgb: 输入的 RGB 图像（numpy 数组）。
            x_offset: 叠加图像在原图中左上角的 x 方向偏移量。
            y_offset: 叠加图像在原图中左上角的 y 方向偏移量。

        返回:
            叠加后的 RGB 图像（numpy 数组）。
        """
        # ----------------------------
        # 1. 生成 matplotlib 图像
        # ----------------------------
        with fig_lock:
            fig, ax = plt.subplots(figsize=(2, 2 ), dpi=200)
            # 设置整个图像背景为透明
            fig.patch.set_alpha(0)
            # 设置坐标区背景为透明（如果需要其他背景颜色，可更改此处）
            ax.set_facecolor((1, 1, 1, 0))

            # 示例：绘制二维高斯分布的等概率线图
            x = np.linspace(-4.0 if xlim is None else -xlim, 4.0 if xlim is None else xlim, 100)
            y = np.linspace(-4.0 if ylim is None else -ylim, 4.0 if ylim is None else ylim, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.exp(-0.5 * (((X - mean_x) / std_x) ** 2 + ((Y - mean_y) / std_y) ** 2))
            contours = ax.contour(X, Y, Z, levels=[0.5, 0.8, 1.0], colors='red')
            # ax.grid(True)
            ax.set_xlim(-4.0 if xlim_axis is None else -xlim_axis, 4.0 if xlim_axis is None else xlim_axis)
            ax.set_ylim(-4.0 if ylim_axis is None else -ylim_axis, 4.0 if ylim_axis is None else ylim_axis)

            # ----------------------------
            # 2. 绘制 Ego 的位置和朝向（等腰三角形 + 朝向箭头）
            # ----------------------------
            angle_offset = np.pi / 12  # 60度的偏移

            ego_triangle = np.array([[ego_x, ego_y],
                                     [ego_x + 0.8 * np.cos(ego_yaw + angle_offset),
                                      ego_y + 0.8 * np.sin(ego_yaw + angle_offset)],
                                     [ego_x + 0.8 * np.cos(ego_yaw - angle_offset),
                                      ego_y + 0.8 * np.sin(ego_yaw - angle_offset)]])
            # ego_triangle = np.array([[ego_x, ego_y],
            #                          [ego_x + 0.1 * np.cos(ego_yaw), ego_y + 0.1 * np.sin(ego_yaw)],
            #                          [ego_x - 0.1 * np.cos(ego_yaw), ego_y - 0.1 * np.sin(ego_yaw)]])
            ax.add_patch(Polygon(ego_triangle, closed=True, facecolor='blue', edgecolor='blue', alpha=0.6))

            # ----------------------------
            # 3. 绘制 Tgt 的位置（五角星标记）
            # ----------------------------
            ax.scatter(tgt_x, tgt_y, color='green', marker='*', s=200, label="Target")
            ax.scatter(mean_x, mean_y, color='red', marker='o', s=20, label="Target")

            # ----------------------------
            # 4. 将 matplotlib 图像保存到内存（使用上下文管理器确保 BytesIO 被关闭）
            # ----------------------------
            with BytesIO() as buf:
                plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                # ----------------------------
                # 5. 读取生成的 matplotlib 图像（含透明通道）
                # ----------------------------
                overlay_img = Image.open(buf).convert("RGBA")
                overlay_img = np.array(overlay_img)
            # 关闭当前 figure，释放内存
            plt.close(fig)
        # ----------------------------
        # 6. 将 matplotlib 图像叠加到原始图像上
        # ----------------------------
        h, w, _ = overlay_img.shape  # 获取 overlay 图像尺寸

        # 确保 ROI 不超过原图尺寸
        if y_offset + h > img_rgb.shape[0] or x_offset + w > img_rgb.shape[1]:
            print(f'叠加图像超出原图边界，请调整偏移量或缩小 overlay 图像尺寸。')
            # return img_rgb
            # raise ValueError("叠加图像超出原图边界，请调整偏移量或缩小 overlay 图像尺寸。")

        # 取出原图的 ROI 区域
        roi = img_rgb[y_offset:y_offset + h, x_offset:x_offset + w]

        # 分离 overlay 图像的 RGB 部分和 alpha 通道
        overlay_rgb = overlay_img[:, :, :3]
        alpha_mask = overlay_img[:, :, 3] / 255.0  # 归一化 alpha 值 [0,1]
        alpha_mask = np.stack([alpha_mask] * 3, axis=2)  # 扩展为三通道

        # 进行 alpha 混合：新的区域 = alpha * overlay + (1 - alpha) * ROI
        blended = (alpha_mask * overlay_rgb + (1 - alpha_mask) * roi).astype(np.uint8)

        # 将混合后的区域覆盖回原图
        img_rgb[y_offset:y_offset + h, x_offset:x_offset + w] = blended[:(min(y_offset + h, img_rgb.shape[0]) - y_offset), :(min(x_offset + w, img_rgb.shape[1]) - x_offset)]

        return img_rgb

    def render_all(self,
                terminal_flag, target_flag=None, ep_return=None,
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
            max_scan_dot = None
            if self._env.cfg.depth.use_camera:
                image_depth = (self._env.depth_buffer[env_id, -1].cpu().numpy() + 1) / 2
                image_depth = np.uint8(255 * image_depth)
                if self._include_depth:
                    overlay_depth_image(image, image_depth, 6)
                images_depth_list[env_id] = image_depth
            else:
                scan_dots = torch.clamp((self._env.measured_heights[env_id] - (-0.8)) / (1.5 - (-0.8)), 0, 1)
                max_scan_dot = scan_dots.max().item()
                scan_dots = scan_dots.reshape((12, 11))
                scan_dots = np.uint8(255 * scan_dots.cpu().numpy())
                scan_dots = scan_dots[::-1]
                scan_dots = scan_dots[:, ::-1]
                if self._include_depth:
                    overlay_depth_image(image, scan_dots, 20)
                # images_depth_list[env_id] = image_depth

            # print(f'{isinstance(self._image, np.ndarray)}, {len(self._image.shape)}')
            # self._image = cv2.UMat(self._image)
            # print(f'image type: {type(self._image)}, image dtype: {self._image.dtype}')
            t = dt * int(self._env.episode_length_buf[env_id])
            text = f'step: {t: .2f}'
            if 'target_std' in kwargs:
                text += f', uncertainty: {kwargs["target_std"][env_id].item(): .3f}, ebd_diff: {kwargs["ebd_diff"][env_id].item(): .3f}'
            if target_flag is not None:
                if target_flag[env_id]:
                    text += ', TGT'
                else:
                    text += ', ORC'
            image = add_text_to_image(image, text)
            if 'manual_std' in kwargs:
                text = f'Set uncertainty: {float(kwargs["manual_std"]): .3f}'
                image = add_text_to_image(image, text, force_add=True)
            text_2 = None
            if ep_return is not None:
                text_2 = f'ep_return: {float(ep_return[env_id].item()): .3f}'
                if max_scan_dot is not None:
                    text_2 += f', max_scan: {float(max_scan_dot): .3f}'
            if text_2 is not None:
                image = add_text_to_image(image, text_2, (10, 75))
            # ego position
            try:
                goal = (self._env.cur_goals[env_id, :2] - self._env.env_origins[env_id, :2] - self._env.base_init_state[:2]).detach().cpu().numpy()
                ego_position = (self._env.root_states[env_id, :2] - self._env.env_origins[env_id, :2] - self._env.base_init_state[:2])
                text_3 = f'x: {float(ego_position[0]): .2f}/{float(goal[0]): .2f}, y: {float(ego_position[1]): .2f}/{float(goal[1]): .2f}, yaw: {float(self._env.yaw[env_id] / 3.14159 * 180.0): .2f}'
                image = add_text_to_image(image, text_3, (10, 115))
            except Exception as e:
                pass
            text_4 = ''
            if 'target_ebd' in kwargs and kwargs['target_ebd'] is not None and 'privileged_ebd' in kwargs and kwargs['privileged_ebd'] is not None and 'noised_embedding' in kwargs and kwargs['noised_embedding'] is not None:
                text_4 = f'ebd: {float(kwargs["target_ebd"][env_id][..., 0]): .2f}/{float(kwargs["privileged_ebd"][env_id][..., 0]): .2f}/{float(kwargs["noised_embedding"][env_id][..., 0]): .2f}'
                if kwargs['privileged_ebd'].shape[-1] > 1:
                    text_4 += f', {float(kwargs["target_ebd"][env_id][..., 1]): .2f}/{float(kwargs["privileged_ebd"][env_id][..., 1]): .2f}/{float(kwargs["noised_embedding"][env_id][..., 1]): .2f}'
            elif 'target_ebd' in kwargs and 'privileged_ebd' in kwargs and kwargs['target_ebd'] is not None and kwargs['privileged_ebd'] is not None:
                text_4 = f'ebd: {float(kwargs["target_ebd"][env_id][..., 0]): .2f}/{float(kwargs["privileged_ebd"][env_id][..., 0]): .2f}'
                if kwargs['privileged_ebd'].shape[-1] > 1:
                    text_4 += f', {float(kwargs["target_ebd"][env_id][..., 1]): .2f}/{float(kwargs["privileged_ebd"][env_id][..., 1]): .2f}'
            elif 'target_ebd' in kwargs and kwargs['target_ebd'] is not None:
                text_4 = f'ebd: {float(kwargs["target_ebd"][env_id][..., 0]): .2f}'
                if kwargs['target_ebd'].shape[-1] > 1:
                    text_4 += f', {float(kwargs["target_ebd"][env_id][..., 1]): .2f}'
            if len(text_4):
                image = add_text_to_image(image, text_4, (10, 155))
            if 'target_logstd' in kwargs and env_id in self._focused_camera_ids[:5]:
                env_class = int(self._env.env_class[env_id])
                if env_class == 17:
                    # plat
                    mean_x = -float(kwargs["target_ebd"][env_id][..., 0]) * 8.0
                    mean_y = -float(kwargs["target_ebd"][env_id][..., 1]) * 8.0
                    std_x = np.exp(float(kwargs["target_logstd"][env_id][..., 0])) * 8.0
                    std_y = np.exp(float(kwargs["target_logstd"][env_id][..., 1])) * 8.0
                    xlim = 4.0
                    ylim = 4.0
                    xlim_axis = 4.0
                    ylim_axis = 4.0
                    yaw = float(self._env.yaw[env_id])
                    ego_x = -float(ego_position[0])
                    ego_y = -float(ego_position[1])
                    tgt_x = -float(goal[0])
                    tgt_y = -float(goal[1])
                elif env_class == 16:
                    # middle_choose
                    if self._env.cfg.depth.use_camera:
                        mean_x = float(kwargs["target_ebd"][env_id][..., 0]) * 2.45 * 4
                        mean_y = float(kwargs["target_ebd"][env_id][..., 1]) * 2.45 * 4
                        std_x = np.exp(float(kwargs["target_logstd"][env_id][..., 0])) * 2.45 * 4
                        std_y = np.exp(float(kwargs["target_logstd"][env_id][..., 1])) * 2.45 * 4
                        xlim = 3.2
                        ylim = 3.2
                        xlim_axis = 4.0
                        ylim_axis = 4.0

                    else:
                        mean_x = float(kwargs["target_ebd"][env_id][..., 0]) * 1.7 * 4
                        mean_y = float(kwargs["target_ebd"][env_id][..., 1]) * 1.7 * 4
                        std_x = np.exp(float(kwargs["target_logstd"][env_id][..., 0])) * 1.7 * 4
                        std_y = np.exp(float(kwargs["target_logstd"][env_id][..., 1])) * 1.7 * 4
                        xlim = 2.5
                        ylim = 2.5
                        xlim_axis = 4.0
                        ylim_axis = 4.0
                    ego_x = float(ego_position[0])
                    ego_y = float(ego_position[1])
                    tgt_x = float(goal[0])
                    tgt_y = float(goal[1])
                    yaw = float(self._env.yaw[env_id]) + np.pi
                elif env_class == 18:
                    # vertical_line
                    mean_x = -float(kwargs["target_ebd"][env_id][..., 0]) * 4.0 * 3.9
                    mean_y = -float(kwargs["target_ebd"][env_id][..., 1]) * 4.0 * 3.9
                    std_x = np.exp(float(kwargs["target_logstd"][env_id][..., 0])) * 4.0 * 3.9
                    std_y = np.exp(float(kwargs["target_logstd"][env_id][..., 1])) * 4.0 * 3.9
                    xlim = 6.0
                    ylim = 4.0
                    xlim_axis = 4.0
                    ylim_axis = 4.0
                    ego_x = -float(ego_position[0])
                    ego_y = -float(ego_position[1])
                    tgt_x = -float(goal[0])
                    tgt_y = -float(goal[1])
                    yaw = float(self._env.yaw[env_id])
                elif env_class == 19:
                    # S
                    mean_x = -float(kwargs["target_ebd"][env_id][..., 0]) * 4.0 * 2.4
                    mean_y = -float(kwargs["target_ebd"][env_id][..., 1]) * 4.0 * 1.55
                    std_x = np.exp(float(kwargs["target_logstd"][env_id][..., 0])) * 4.0 * 2.4
                    std_y = np.exp(float(kwargs["target_logstd"][env_id][..., 1])) * 4.0 * 1.55
                    xlim = 4.0
                    ylim = 4.0
                    xlim_axis = 4.0
                    ylim_axis = 4.0
                    ego_x = -float(ego_position[0])
                    ego_y = -float(ego_position[1])
                    tgt_x = -float(goal[0])
                    tgt_y = -float(goal[1])
                    yaw = float(self._env.yaw[env_id])
                elif env_class == 20:
                    # middle_choose
                    mean_x = float(kwargs["target_ebd"][env_id][..., 0]) * 2.0 * 4
                    mean_y = float(kwargs["target_ebd"][env_id][..., 1]) * 2.0 * 4
                    std_x = np.exp(float(kwargs["target_logstd"][env_id][..., 0])) * 2.0 * 4
                    std_y = np.exp(float(kwargs["target_logstd"][env_id][..., 1])) * 2.0 * 4
                    xlim = 2.5
                    ylim = 2.5
                    xlim_axis = 4.0
                    ylim_axis = 4.0

                    ego_x = float(ego_position[0])
                    ego_y = float(ego_position[1])
                    tgt_x = float(goal[0])
                    tgt_y = float(goal[1])
                    yaw = float(self._env.yaw[env_id]) + np.pi

                # mean_x = -float(kwargs["target_ebd"][env_id][..., 0]) * 2.0
                self.overlay_matplotlib_image(image, x_offset=int(image.shape[1] * 0.75), y_offset=int(image.shape[0] * 0.6),
                                              mean_x=mean_x,
                                              mean_y=mean_y,
                                              std_x=std_x,
                                              std_y=std_y,
                                              ego_x=ego_x,
                                              ego_y=ego_y,
                                              ego_yaw=yaw,
                                              tgt_x=tgt_x,
                                              tgt_y=tgt_y,
                                              xlim=xlim,
                                              ylim=ylim,
                                              fig_lock=self.fig_lock,
                                              xlim_axis=xlim_axis,
                                              ylim_axis=ylim_axis
                                              )
            root_pos = self._env.root_states[env_id, :3].cpu().numpy()
            # cam_pos = root_pos + self.cam_pos_rel
            root_pos = (self._env.env_origins[env_id, :3] + self._env.base_init_state[:3]).cpu().numpy()
            cam_pos = root_pos + self.cam_pos_rel

            # OK below
            if self._view_mode == 'plat':
                cam_pos = root_pos + self.cam_pos_rel * 0.7
                root_pos[2] -= 4
                root_pos[0] += 1
                cam_pos[0] += 2
                # root_pos[1] -= 2
                # cam_pos[2]
            # end
            # 侧视图
            # cam_pos_rel = self.cam_pos_rel * 0.6
            # cam_pos_rel[0], cam_pos_rel[1] = cam_pos_rel[1], cam_pos_rel[0]
            # cam_pos = root_pos + cam_pos_rel
            # cam_pos[2] += 0.3
            # root_pos[2] -= 4
            # root_pos[0] -= 2
            # 侧视图end
            # 反向视图
            if self._view_mode == 'square':
                cam_pos_rel = self.cam_pos_rel * 0.7
                cam_pos_rel[0], cam_pos_rel[1] = -cam_pos_rel[0], -cam_pos_rel[1]
                cam_pos = root_pos + cam_pos_rel
                cam_pos[2] -= 0.4
                cam_pos[0] -= 0.8
                root_pos[0] += 0.8
                root_pos[2] -= 4
                root_pos[1] += 2

            if self._view_mode == 'middle_choose':
                cam_pos_rel = self.cam_pos_rel * 0.7
                cam_pos_rel[0], cam_pos_rel[1] = -cam_pos_rel[0], -cam_pos_rel[1]
                cam_pos = root_pos + cam_pos_rel
                cam_pos[2] -= 0.4
                cam_pos[0] -= 0.8
                root_pos[0] += 0.8
                root_pos[2] -= 4
                root_pos[1] += 2
                height = cam_pos[2]
                cam_pos_rel = cam_pos - root_pos
                cam_pos_rel = cam_pos_rel * 0.8
                cam_pos = cam_pos_rel + root_pos
                cam_pos[2] = height

            if self._view_mode == 'four_corners':
                cam_pos = root_pos + self.cam_pos_rel * 0.7
            if self._view_mode == 'left_right_choose':
                cam_pos = root_pos + self.cam_pos_rel * 0.6

            # 反向视图end



            # env_origins = self._env.env_origins[env_id, :3].cpu().numpy()
            # if env_id == 0:
            #     print(f'cam_pos: {cam_pos}, root_pos: {root_pos}')
            self._gym.set_camera_location(self._cameras[env_id], self._envs[env_id],
                                          gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
            image_list[env_id] = image

            return env_id
        with ThreadPoolExecutor(max_workers=15) as executor:
            results = list(executor.map(modify_image_i, self._focused_camera_ids))
        self._images_list = image_list
        self._image = self._images_list[self._camera_id]

        def write_video(env_id):
            video_name = f'env_{env_id:02}.mp4'
            attachment_name = f'env_{env_id:02}.pkl'
            if self._attachment_buffers[env_id] is None:
                self._attachment_buffers[env_id] = []
            if 'save_attachments' in kwargs:
                self._attachment_buffers[env_id].append({k: v[env_id].cpu().detach() for k, v in kwargs['save_attachments'].items()})
            if terminal_flag[env_id]:
                if self._video_writers[env_id] is not None:
                    self._video_writers[env_id].release()
                    self._video_writers[env_id] = None
                    if os.path.exists(os.path.join(self.complete_dir, video_name)):
                        os.remove(os.path.join(self.complete_dir, video_name))
                    shutil.move(os.path.join(self.processing_dir, video_name), os.path.join(self.complete_dir, video_name))
                if self._attachment_buffers[env_id] is not None:
                    if os.path.exists(os.path.join(self.attachment_dir, attachment_name)):
                        os.remove(os.path.join(self.attachment_dir, attachment_name))
                    with open(os.path.join(self.attachment_dir, attachment_name), 'wb') as f:
                        pickle.dump(self._attachment_buffers[env_id], f)
                    self._attachment_buffers[env_id] = [self._attachment_buffers[env_id][-1]] if len(self._attachment_buffers[env_id]) > 1 else None

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式编码
            dir_name = self.processing_dir
            frame_height, frame_width, _ = self._images_list[env_id].shape
            new_size = (int(frame_width // self.resize_ratio), int(frame_height // self.resize_ratio))
            if self._video_writers[env_id] is None:
                self._video_writers[env_id] = cv2.VideoWriter(os.path.join(dir_name, video_name), fourcc, 1/dt, new_size)

            img_bgr = cv2.cvtColor(self._images_list[env_id], cv2.COLOR_RGB2BGR)
            frame_height, frame_width, _ = img_bgr.shape
            img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)
            self._video_writers[env_id].write(img_bgr)
        with ThreadPoolExecutor(max_workers=15) as executor:
            results = list(executor.map(write_video, self._focused_camera_ids))

        if self._env.cfg.depth.use_camera:
            self._image_depth = images_depth_list[self._camera_id]
    def close(self):
        for video_writer in self._video_writers:
            if video_writer is not None:
                video_writer.release()


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
