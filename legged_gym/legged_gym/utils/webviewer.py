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


def overlay_depth_image(image, image_depth):
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
    scale_factor = 2
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

def quat2mat(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    把四元数 (x, y, z, w) 转成 3×3 旋转矩阵
    参数:
        quat_xyzw: shape (4,) 的 ndarray，顺序是 (x, y, z, w)
    返回:
        R: shape (3,3) 的旋转矩阵
    """
    x, y, z, w = quat_xyzw
    # 预计算乘积
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ], dtype=np.float32)
    return R


def world_to_pixel(point_world, view_mat, proj_mat, image_width, image_height):
    """
    point_world: (3,) 或 (x, y, z)
    view_mat:   (4,4) world->camera 变换（你从 IsaacGym 拿到的 view_mat.T）
    proj_mat:   (4,4) camera->clip 投影矩阵（你从 IsaacGym 拿到的 proj_mat.T）
    返回: (u, v, depth_ndc)
    """
    # 1. 齐次坐标
    pw = np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float32)

    # 2. 世界 -> 相机
    pc = view_mat @ pw  # shape (4,)

    # 3. 相机 -> 裁剪
    p_clip = proj_mat @ pc  # shape (4,)

    # 4. 透视除法得到 NDC
    ndc = p_clip[:3] / p_clip[3]  # shape (3,)

    # 5. NDC -> 像素
    u = (ndc[0] + 1.0) * 0.5 * image_width
    v = (1.0 - ndc[1]) * 0.5 * image_height

    return np.array([u, v, ndc[2]], dtype=np.float32)

#
# def world_to_pixel(P_w, K, R, t, check_valid=True):
#     """
#     将 3D 世界点投影到像素坐标系
#     ---------------------------------------------------
#     P_w : np.ndarray shape (3,)   # 世界坐标 (x,y,z)
#     K   : np.ndarray shape (3,3)  # 内参矩阵
#     R   : np.ndarray shape (3,3)  # 世界 -> 相机 旋转
#     t   : np.ndarray shape (3,)   # 平移向量
#     check_valid : bool            # 是否丢弃在相机后方/视锥外的点
#     ---------------------------------------------------
#     return: (u,v,depth) 或 None
#     """
#     # 1. 变换到相机坐标系
#     P_c = R @ P_w + t  # shape (3,)
#     print(f'P_c: {P_c}')
#     # 2. 可选：深度有效性检查
#     z = P_c[2]
#     if check_valid and z <= 1e-6:  # 这里假设相机坐标系 +Z 指向成像平面
#         return None  # 如果你的 z_cam 定义为 -Z，则改成 z >= -1e-6
#
#     # 3. 像素坐标（齐次归一化）
#     p_h = K @ P_c  # (u*depth, v*depth, depth)
#     u = p_h[0] / p_h[2]
#     v = p_h[1] / p_h[2]
#
#     return np.array([u, v, z])

class WebViewer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555) -> None:
        """
        Web viewer for Isaac Gym

        :param host: Host address (default: "127.0.0.1")
        :type host: str
        :param port: Port number (default: 5000)
        :type port: int
        """
        self._app = flask.Flask(__name__)
        self._app.add_url_rule("/", view_func=self._route_index)
        self._app.add_url_rule("/_route_stream", view_func=self._route_stream)
        self._app.add_url_rule("/_route_stream_depth", view_func=self._route_stream_depth)
        self._app.add_url_rule("/_route_input_event", view_func=self._route_input_event, methods=["POST"])

        self._log = logging.getLogger('werkzeug')
        self._log.disabled = True
        self._app.logger.disabled = True

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

        # start server
        self._thread = threading.Thread(target=lambda: \
            self._app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True)
        self._thread.start()
        print(f"\nStarting web viewer on http://{host}:{port}/\n")

    def _route_index(self) -> 'flask.Response':
        """Render the web page

        :return: Flask response
        :rtype: flask.Response
        """
        with open(os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "utils", "webviewer.html"), 'r', encoding='utf-8') as file:
            template = file.read()
        self._event_load.set()
        return flask.render_template_string(template)

    def _route_stream(self) -> 'flask.Response':
        """Stream the image to the web page

        :return: Flask response
        :rtype: flask.Response
        """
        return flask.Response(self._stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _route_stream_depth(self) -> 'flask.Response':
        return flask.Response(self._stream_depth(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _route_input_event(self) -> 'flask.Response':

        # get keyboard and mouse inputs
        data = flask.request.get_json()
        key, mouse = data.get("key", None), data.get("mouse", None)
        dx, dy, dz = data.get("dx", None), data.get("dy", None), data.get("dz", None)

        transform = self._gym.get_camera_transform(self._sim,
                                                   self._envs[self._camera_id],
                                                   self._cameras[self._camera_id])

        # zoom in/out
        if mouse == "wheel":
            # compute zoom vector
            r, theta, phi = cartesian_to_spherical(*self.cam_pos_rel)
            r += 0.05 * dz
            self.cam_pos_rel = spherical_to_cartesian(r, theta, phi)

        # orbit camera
        elif mouse == "left":
            # convert mouse movement to angle
            dx *= 0.2 * math.pi / 180
            dy *= 0.2 * math.pi / 180

            r, theta, phi = cartesian_to_spherical(*self.cam_pos_rel)
            theta -= dy
            phi -= dx
            self.cam_pos_rel = spherical_to_cartesian(r, theta, phi)

        # pan camera
        elif mouse == "right":
            # convert mouse movement to angle
            dx *= -0.2 * math.pi / 180
            dy *= -0.2 * math.pi / 180

            r, theta, phi = cartesian_to_spherical(*self.cam_pos_rel)
            theta += dy
            phi += dx
            self.cam_pos_rel = spherical_to_cartesian(r, theta, phi)

        elif key == 219:  # prev
            self._camera_id = (self._camera_id-1) % self._env.num_envs
            return flask.Response(status=200)
        
        elif key == 221:  # next
            self._camera_id = (self._camera_id+1) % self._env.num_envs
            return flask.Response(status=200)
        
        # pause stream (V: 86)
        elif key == 86:
            self._pause_stream = not self._pause_stream
            return flask.Response(status=200)

        # change image type (T: 84)
        elif key == 84:
            if self._camera_type == gymapi.IMAGE_COLOR:
                self._camera_type = gymapi.IMAGE_DEPTH
            elif self._camera_type == gymapi.IMAGE_DEPTH:
                self._camera_type = gymapi.IMAGE_COLOR
            return flask.Response(status=200)
        elif key == 113 or key == 81:
            self._stopped = True
            return flask.Response(status=200)
        else:
            return flask.Response(status=200)

        return flask.Response(status=200)

    def _stream(self) -> bytes:
        """Format the image to be streamed

        :return: Image encoded as Content-Type
        :rtype: bytes
        """
        while True:
            self._event_stream.wait()

            # prepare image
            image = imageio.imwrite("<bytes>", self._image, format="JPEG")

            # stream image
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

            self._event_stream.clear()
            self._notified = False

    def _stream_depth(self) -> bytes:
        while self._env.cfg.depth.use_camera:
            self._event_stream_depth.wait()

            # prepare image
            image = imageio.imwrite("<bytes>", self._image_depth, format="JPEG")

            # stream image
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

            self._event_stream_depth.clear()

    @staticmethod
    def get_inner_parameter(camera_props):
        # camera_props: gymapi.CameraProperties
        hfov = camera_props.horizontal_fov
        print(f'horizontal_fov: {hfov}')
        W, H = camera_props.width, camera_props.height
        vfov = hfov * H / W  # 文档说明垂直 FOV = 高/宽 × horizontal_fov

        fx = W / (2 * np.tan(hfov / 2))
        fy = H / (2 * np.tan(vfov / 2))
        cx = W / 2
        cy = H / 2

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])

        return K

    @staticmethod
    def get_outer_parameter(cam_pos, root_pos):
        # cam_pos: (3, ), np.ndarray
        # root_pos: (3, ), np.ndarray
        u_world = np.array([0, 0, 1])
        z_cam = -(cam_pos - root_pos)
        z_cam /= np.linalg.norm(z_cam)  # 摄像机坐标系的 -Z 轴（朝向负光轴方向）
        x_cam = np.cross(u_world, z_cam)
        x_cam /= np.linalg.norm(x_cam)  # 摄像机坐标系的 X 轴
        y_cam = np.cross(z_cam, x_cam)  # 摄像机坐标系的 Y 轴
        print(f'cam_pos: {cam_pos}, root_pos: {root_pos}, z_cam: {z_cam}, x_cam: {x_cam}, y_cam: {y_cam}')
        R = np.stack([x_cam, y_cam, z_cam], axis=0)  # 世界->相机的旋转
        t = -R @ cam_pos  # 平移
        return R, t, root_pos

    def attach_view_camera(self, i, env_handle, actor_handle, root_pos):
        if True:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 960
            camera_props.height = 540
            # camera_props.enable_tensors = True
            # camera_props.horizontal_fov = camera_horizontal_fov
            # horizontal_fov（单位：弧度）在 camera_props.horizontal_fov

            camera_handle = self._gym.create_camera_sensor(env_handle, camera_props)
            # self._camera_k[i] = self.get_inner_parameter(camera_props)
            self._cameras.append(camera_handle)
            
            cam_pos = root_pos + np.array([0, 1, 0.5])
            self._gym.set_camera_location(camera_handle, env_handle, gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
            # self._camera_Rt[i] = self.get_outer_parameter(cam_pos, root_pos)

    @staticmethod
    def add_square_to_image(point_world, root_pos, view_mat, proj_mat, image):
        # relative_pos = point_world - root_pos
        # relative_pos[0] = -relative_pos[0]
        # relative_pos[1] = relative_pos[1]
        # relative_pos[2] = 0.0
        # point_world = root_pos + relative_pos
        # ——— 2. 投影到像素平面 ———
        H, W = image.shape[:2]
        # point_world, view_mat, proj_mat, image_width, image_height):
        uv_depth = world_to_pixel(point_world, view_mat, proj_mat, H, W)
        print(f'uv_depth: {uv_depth}')
        # print(f'point_world: {point_world}, uv_depth: {uv_depth}, K: {K}, R: {R}, t: {t}')
        if uv_depth is not None:
            u, v, _ = uv_depth


            u, v = int(round(u)), int(round(v))
            # u, v = v, u
            # 边界检查
            if 0 <= u < W and 0 <= v < H:
                # ——— 3. 在像素图上画 50×50 的绿色框 ———
                half = 25
                pt1 = (u - half, v - half)
                pt2 = (u + half, v + half)
                cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
            else:
                print(f"Warning: point {point_world} is out of image bounds, {(u, v)}")
        else:
            print(f'uv_depth is None, point_world: {point_world}, K: {K}, R: {R}, t: {t}')

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
        self.cam_pos_rel = np.array([0, 2, 1])
        # self._camera_k = []
        # self._camera_Rt = []
        for i in range(self._env.num_envs):
            # self._camera_k.append(None)
            # self._camera_Rt.append(None)
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
        # wait for page to load
        if self._wait_for_page:
            if wait_for_page_load:
                if not self._event_load.is_set():
                    print("Waiting for web page to begin loading...")
                self._event_load.wait()
                self._event_load.clear()
            self._wait_for_page = False

        # pause stream
        if self._pause_stream:
            return

        if self._notified:
            return

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
        # point_world = self._env.cur_goals[self._camera_id, :3].detach().cpu().numpy()
        #
        # point_world[2] = 0
        # view_mat = np.array(
        #     self._gym.get_camera_view_matrix(self._sim, self._envs[self._camera_id],  self._cameras[self._camera_id])
        # ).reshape((4, 4)).T
        # proj_mat = np.array(
        #     self._gym.get_camera_proj_matrix(self._sim, self._envs[self._camera_id],  self._cameras[self._camera_id])
        # ).reshape((4, 4)).T
        # print(f'view_mat: {view_mat}, proj_mat: {proj_mat}')
        # self.add_square_to_image(
        #     # point_world, root_pos, view_mat, proj_mat, image
        #     point_world,
        #     self._camera_Rt[self._camera_id][2],
        #     view_mat, proj_mat,
        #     self._image
        # )
        root_pos = self._env.root_states[self._camera_id, :3].cpu().numpy()
        cam_pos = root_pos + self.cam_pos_rel
        self._gym.set_camera_location(self._cameras[self._camera_id], self._envs[self._camera_id], gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
        # self._camera_Rt[self._camera_id] = self.get_outer_parameter(cam_pos, root_pos)


        # notify stream thread
        self._event_stream.set()
        if self._env.cfg.depth.use_camera:
            self._event_stream_depth.set()
        self._notified = True

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
        # wait for page to load
        if self._wait_for_page:
            if wait_for_page_load:
                if not self._event_load.is_set():
                    print("Waiting for web page to begin loading...")
                self._event_load.wait()
                self._event_load.clear()
            self._wait_for_page = False

        # pause stream
        if self._pause_stream:
            return

        if self._notified:
            return

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

            # print(f'{isinstance(self._image, np.ndarray)}, {len(self._image.shape)}')
            # self._image = cv2.UMat(self._image)
            # print(f'image type: {type(self._image)}, image dtype: {self._image.dtype}')
            t = dt * int(self._env.episode_length_buf[env_id])
            text = f'step: {t: .2f}'
            if 'target_std' in kwargs:
                text += f', uncertainty: {kwargs["target_std"][env_id].item():.3f}, ebd_diff: {kwargs["ebd_diff"][env_id].item():.3f}'
            image = add_text_to_image(image, text)

            root_pos = self._env.root_states[env_id, :3].cpu().numpy()
            cam_pos = root_pos + self.cam_pos_rel
            self._gym.set_camera_location(self._cameras[env_id], self._envs[env_id],
                                          gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos))
            # self._camera_Rt[env_id] = self.get_outer_parameter(cam_pos, root_pos)
            image_list[env_id] = image
            return env_id
        with ThreadPoolExecutor(max_workers=15) as executor:
            results = list(executor.map(modify_image_i, range(self._env.num_envs)))
        self._images_list = image_list
        self._image = self._images_list[self._camera_id]
        if self._env.cfg.depth.use_camera:
            self._image_depth = images_depth_list[self._camera_id]
        # notify stream thread
        self._event_stream.set()
        if self._env.cfg.depth.use_camera:
            self._event_stream_depth.set()
        self._notified = True

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
