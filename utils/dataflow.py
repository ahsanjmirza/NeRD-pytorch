import os
import imageio
import numpy as np
import PIL.Image
from utils.exif_helper import formatted_exif_data, getExposureTime
from utils.exposure_helper import calculate_ev100_from_metadata
from skimage.transform import rescale


def _minify(basedir, factor):
    """
    Minifies the images and masks in the specified directory by the given factor.

    Args:
        basedir (str): The base directory containing the images and masks.
        factor (int): The factor by which to minify the images and masks.

    Returns:
        None
    """

    imgdir = os.path.join(basedir, "images_{}".format(factor))
    maskdir = os.path.join(basedir, "masks_{}".format(factor))

    imgdir = os.path.join(basedir, "images")
    maskdir = os.path.join(basedir, "masks")
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    masks = [os.path.join(maskdir, f) for f in sorted(os.listdir(maskdir))]
    imgs = [
        f
        for f in imgs
        if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
    ]
    masks = [
        f
        for f in masks
        if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
    ]
    imgdir_orig = imgdir
    maskdir_orig = maskdir

    nameImg = "images_{}".format(factor)
    nameMask = "masks_{}".format(factor)
    imgdir = os.path.join(basedir, nameImg)
    maskdir = os.path.join(basedir, nameMask)

    if not os.path.exists(imgdir):
        os.mkdir(imgdir)
    if not os.path.exists(maskdir):
        os.mkdir(maskdir)
        
    print("Minifying", factor, basedir)

    for f in os.listdir(imgdir_orig):
        if not os.path.exists(os.path.join(imgdir, f)):
            img = rescale(imageio.imread(os.path.join(imgdir_orig, f)), scale=1/factor, channel_axis=-1, preserve_range=True)
            imageio.imwrite(os.path.join(imgdir, f), np.uint8(img))

    for f in os.listdir(maskdir_orig):
        if not os.path.exists(os.path.join(maskdir, f)):
            img = rescale(imageio.imread(os.path.join(maskdir_orig, f)), scale=1/factor, channel_axis=-1, preserve_range=True)
            imageio.imwrite(os.path.join(maskdir, f), np.uint8(img))

    return

def ensureNoChannel(x: np.ndarray) -> np.ndarray:
    """
    Ensure that the input array has no channel dimension.

    If the input array has more than 2 dimensions, the channel dimension is removed by selecting the first channel.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The modified array with no channel dimension.
    """
    ret = x
    if len(x.shape) > 2:
        ret = ret[:, :, 0]

    return ret


def _load_data(basedir, factor=1, load_imgs=True):
    """
    Load data from the specified directory.

    Args:
        basedir (str): The base directory path.
        factor (int, optional): The factor to apply to the data. Defaults to 1.
        load_imgs (bool, optional): Whether to load images or not. Defaults to True.

    Returns:
        tuple: A tuple containing the loaded data:
            - poses (ndarray): The poses array.
            - bds (ndarray): The bounds array.
            - imgs (ndarray): The images array.
            - masks (ndarray): The masks array.
            - ev100s (list): The EV100 values.
    """
    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [
        os.path.join(basedir, "images", f)
        for f in sorted(os.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]
    sh = imageio.imread(img0).shape

    ev100s = handle_exif(basedir)

    sfx = ""

    if factor != 1:
        print("Apply factor", factor)
        sfx = "_{}".format(factor)
        _minify(basedir, factor=factor)
        factor = factor

    imgdir = os.path.join(basedir, "images" + sfx)
    maskdir = os.path.join(basedir, "masks" + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, "does not exist, returning")
        return

    if not os.path.exists(maskdir):
        print(maskdir, "does not exist, returning")
        return

    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]

    masksfiles = [
        os.path.join(maskdir, f)
        for f in sorted(os.listdir(maskdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    
    if poses.shape[-1] != len(imgfiles) and len(imgfiles) == len(masksfiles):
        print(
            "Mismatch between imgs {}, masks {} and poses {} !!!!".format(
                len(imgfiles), len(masksfiles), poses.shape[-1]
            )
        )
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith("png"):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255.0 for f in imgfiles]
    masks = [ensureNoChannel(imread(f) / 255.0) for f in masksfiles]

    imgs = np.stack(imgs, -1)
    masks = np.stack(masks, -1)

    return poses, bds, imgs, masks, ev100s


def handle_exif(basedir):
    """
    Handle EXIF data for a given directory of images.

    Args:
        basedir (str): The base directory containing the images.

    Returns:
        numpy.ndarray: A numpy array containing the EV100 values calculated from the EXIF data of the images.

    Raises:
        Exception: If no JPEG images are found in the directory.
        Exception: If the number of valid images does not match the total number of images.

    """
    files = [
        os.path.join(basedir, "images", f)
        for f in sorted(os.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg")
    ]

    if len(files) == 0:
        raise Exception(
            "Only jpegs can be used to gain EXIF data needed to handle tonemapping"
        )

    ret = []
    for f in files:
        try:
            img = PIL.Image.open(f)
            exif_data = formatted_exif_data(img)

            iso = int(exif_data["ISOSpeedRatings"])
            aperture = float(exif_data["FNumber"])

            exposureTime = getExposureTime(exif_data)

            ev100 = calculate_ev100_from_metadata(aperture, exposureTime, iso)
        except:
            ev100 = 8

        ret.append(ev100)

    if len(files) != len(ret):
        raise Exception("Not all images are valid!")

    return np.stack(ret, 0).astype(np.float32)


import numpy as np

def normalize(x):
    """
    Normalize a vector.

    Parameters:
    x (numpy.ndarray): The input vector.

    Returns:
    numpy.ndarray: The normalized vector.
    """
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    """
    Calculates the view matrix for a given camera position and orientation.

    Args:
        z (array-like): The direction vector of the camera's line of sight.
        up (array-like): The up vector of the camera.
        pos (array-like): The position of the camera.

    Returns:
        array-like: The view matrix.

    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    """
    Transforms 3D points from world coordinates to camera coordinates.

    Args:
        pts (numpy.ndarray): Array of shape (N, 3) representing the 3D points in world coordinates.
        c2w (numpy.ndarray): 4x4 transformation matrix representing the camera-to-world transformation.

    Returns:
        numpy.ndarray: Array of shape (N,) representing the transformed points in camera coordinates.
    """
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    """
    Calculate the average camera-to-world transformation matrix.

    Args:
        poses (numpy.ndarray): Array of camera-to-world transformation matrices.

    Returns:
        numpy.ndarray: Average camera-to-world transformation matrix.
    """

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    """
    Generates a list of camera poses for rendering a spiral path.

    Args:
        c2w (numpy.ndarray): Camera-to-world transformation matrix.
        up (numpy.ndarray): Up vector of the camera.
        rads (list): List of radii for each spiral loop.
        focal (float): Focal length of the camera.
        zdelta (float): Delta value for z-axis translation.
        zrate (float): Rate of change for z-axis translation.
        rots (int): Number of rotations around the spiral path.
        N (int): Number of camera poses to generate.

    Returns:
        list: List of camera poses represented as transformation matrices.

    """
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):
    """
    Recenter the poses by aligning them to the average pose.

    Args:
        poses (np.ndarray): Array of shape (N, 3, 4) representing camera poses.

    Returns:
        np.ndarray: Array of shape (N, 3, 4) with recentered camera poses.
    """
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    """
    Spherify the poses of a camera trajectory.

    Args:
        poses (numpy.ndarray): Array of shape (N, 3, 4) representing the camera poses.
        bds (numpy.ndarray): Array of shape (N, 2) representing the bounding boxes.

    Returns:
        numpy.ndarray: Array of shape (N, 3, 4) representing the spherified camera poses.
        numpy.ndarray: Array of shape (N, 4, 4) representing the new spherified camera poses.
        numpy.ndarray: Array of shape (N, 2) representing the updated bounding boxes.
    """
    def p34_to_44(p):
        return np.concatenate(
            [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
        )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def load_llff_data(
    basedir, factor=8, recenter=True, bd_factor=0.75, spherify=False, path_zflat=False
):
    """
    Load LLFF data from the specified directory.

    Args:
        basedir (str): The base directory path.
        factor (int, optional): The downsampling factor. Defaults to 8.
        recenter (bool, optional): Whether to recenter the poses. Defaults to True.
        bd_factor (float, optional): The bounding box factor for rescaling. Defaults to 0.75.
        spherify (bool, optional): Whether to spherify the poses. Defaults to False.
        path_zflat (bool, optional): Whether to flatten the z-axis of the path. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - images (ndarray): The images.
            - masks (ndarray): The masks.
            - ev100s (ndarray): The exposure values.
            - poses (ndarray): The camera poses.
            - bds (ndarray): The bounding boxes.
            - render_poses (ndarray): The poses for rendering.
    """
    poses, bds, imgs, msks, ev100s = _load_data(
        basedir, factor=factor,
    )  

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    msks = np.expand_dims(np.moveaxis(msks, -1, 0), -1).astype(np.float32)
    images = imgs
    masks = msks
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        c2w = poses_avg(poses)

        # Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.0
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=N_rots, N=N_views
        )

    render_poses = np.array(render_poses).astype(np.float32)

    images = images.astype(np.float32)
    masks = masks.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, masks, ev100s, poses, bds, render_poses
