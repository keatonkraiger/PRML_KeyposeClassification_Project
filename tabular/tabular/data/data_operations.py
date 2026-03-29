import numpy as np
from scipy.spatial.transform import Rotation as R

#------------- Data manipulation -------------#
def center_pose(pos, center_joint):
    """Center pose data around the center_joint, supports 2D or 3D + confidence."""
    return pos - pos[:, center_joint:center_joint+1, :]

def convert_to_quaternion(pose_data, center_joint):
    """
    Convert (N, J, D+1) pose data (last dim = confidence) to (N, J, 4) quaternions.
    Only supports 3D.
    """
    pos = pose_data[..., :-1]  # (N, J, D)
    conf = pose_data[..., -1]
    N, J, D = pos.shape

    if D != 3:
        raise ValueError(f"Quaternion conversion requires 3D input, got {D}D")

    quats = np.zeros((N, J, 4))
    for i in range(N):
        for j in range(J):
            if j == center_joint or conf[i, j] == 0 or np.linalg.norm(pos[i, j]) < 1e-8:
                quats[i, j] = [1, 0, 0, 0]  # Identity quaternion
                continue

            joint_vec = pos[i, j] / np.linalg.norm(pos[i, j])
            up = np.array([0, 1, 0]) if abs(np.dot(joint_vec, [0, 1, 0])) < 0.99 else np.array([0, 0, 1])
            right = np.cross(up, joint_vec)
            right /= np.linalg.norm(right)
            up_corrected = np.cross(joint_vec, right)
            up_corrected /= np.linalg.norm(up_corrected)

            R_mat = np.stack([joint_vec, up_corrected, np.cross(joint_vec, up_corrected)], axis=1)
            quat = R.from_matrix(R_mat).as_quat()  # (x, y, z, w)
            quats[i, j] = [quat[3], quat[0], quat[1], quat[2]]  # Convert to (w, x, y, z)
    return quats

def convert_to_euler(pose_data, center_joint):
    """
    Convert (N, J, D+1) pose data (last dim = confidence) to (N, J, 3) Euler angles.
    Only supports 3D.
    """
    pos = pose_data[..., :-1]  
    conf = pose_data[..., -1]
    N, J, D = pos.shape

    if D != 3:
        raise ValueError(f"Euler conversion requires 3D input, got {D}D")

    eulers = np.zeros((N, J, 3))
    for i in range(N):
        for j in range(J):
            if j == center_joint or conf[i, j] == 0 or np.linalg.norm(pos[i, j]) < 1e-8:
                eulers[i, j] = [0, 0, 0]
                continue

            joint_vec = pos[i, j] / np.linalg.norm(pos[i, j])
            up = np.array([0, 1, 0]) if abs(np.dot(joint_vec, [0, 1, 0])) < 0.99 else np.array([0, 0, 1])
            right = np.cross(up, joint_vec)
            right /= np.linalg.norm(right)
            up_corrected = np.cross(joint_vec, right)
            up_corrected /= np.linalg.norm(up_corrected)

            R_mat = np.stack([joint_vec, up_corrected, np.cross(joint_vec, up_corrected)], axis=1)
            eulers[i, j] = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
    return eulers


#------------- Data normalization functions -------------#
def normalize_pressure_dist(data):
    """Normalize pressure maps into per-frame probability distributions.

    Expects pressure data where the last two dimensions are spatial/foot
    dimensions (e.g. (N, T, 60, 21, 2) or (N, 60, 21, 2)). Each frame is
    normalized by the sum over those last two dimensions so that every
    per-frame map sums to 1, instead of normalizing by the total over the
    entire sequence.
    """
    data = np.asarray(data, dtype=np.float32)

    # Sum over spatial + foot dimensions only, keep leading dims (N, T, ...).
    totals = np.nansum(data, axis=(1, 2, 3), keepdims=True)

    # Avoid divide-by-zero: frames with non-positive total are left unchanged.
    safe_totals = np.where(totals > 0, totals, 1.0)
    normalized = data / safe_totals
    return np.where(totals > 0, normalized, data)

def normalize_pressure_max(data, max_val):
    """Divide each frame by a subject-wise max pressure value."""
    return data / (max_val + 1e-8)

def normalize_pressure_log(data, max_val):
    """Apply a log transform to pressure data, scaled by the max pressure."""
    return np.log1p(data) / np.log1p(max_val)

def log(data):
    """Element-wise log transform with basic numerical safety.

    Not used by default in the student pipeline, but kept as a simple
    example of a non-linear transform that students can opt into if they
    decide to implement their own log-based normalization.
    """
    np.seterr(divide='ignore', invalid='ignore')
    # Clamp to avoid log(0) and negative values causing NaNs.
    data = np.asarray(data, dtype=np.float32)
    safe = np.clip(data, a_min=1e-8, a_max=None)
    return np.log(safe)

def compute_norm_stats(data):
    """Compute mean and std across (N, J, D) pose data"""
    flat = data.reshape(-1, data.shape[-1])
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    return mean, std

def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)


def minmax(data, min_vals=None, diff_vals=None):
    """Feature-wise min-max normalization.

    Mirrors the behavior expected by the tabular pipeline and is used
    when ``norm: MINMAX`` is selected in the config.
    """
    np.seterr(invalid='ignore')
    data = np.asarray(data, dtype=np.float32)

    if min_vals is None or diff_vals is None:
        max_vals = np.max(data, axis=0)
        min_vals = np.min(data, axis=0)
        diff_vals = max_vals - min_vals

    norm_data = (data - min_vals) / diff_vals
    norm_data = np.nan_to_num(norm_data, nan=0.0)
    return norm_data, min_vals, diff_vals


def zscore(data, mean_vals=None, std_vals=None):
    """Feature-wise z-score normalization.

    This is the default normalization used for mocap_3d when
    ``norm: ZSCORE`` is set in the config.
    """
    np.seterr(invalid='ignore')
    data = np.asarray(data, dtype=np.float32)

    if mean_vals is None or std_vals is None:
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)

    norm_data = (data - mean_vals) / std_vals
    norm_data = np.nan_to_num(norm_data, nan=0.0)
    return norm_data, mean_vals, std_vals

#------------- LOD (Level of Detail) Downsampling for Pressure -----#
def downsample_pressure_frame(frame):
    """
    Downsample a 2D pressure/force map by summing 2x2 grids.
    This preserves the total force/pressure sum (important for distribution maps).
    
    Args:
        frame: (H, W) array
        
    Returns:
        (ceil(H/2), ceil(W/2)) downsampled array
    """
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame, got shape {frame.shape}")

    # reduceat naturally handles odd dimensions by creating a final 1x2, 2x1,
    # or 1x1 block; this keeps the total sum exactly preserved.
    row_idx = np.arange(0, frame.shape[0], 2)
    col_idx = np.arange(0, frame.shape[1], 2)
    return np.add.reduceat(np.add.reduceat(frame, row_idx, axis=0), col_idx, axis=1)


def apply_lod(pressure_data, lod_level):
    """
    Apply Level of Detail (LOD) downsampling to pressure data.
    
    This function takes foot pressure maps and progressively downsamples them while
    maintaining the overall force/pressure totals (important for distribution maps 
    where sum across all sensors must be preserved).
    
    IMPORTANT: Pressure data must be in spatial form (N, H, W, 2), not flattened.
    Ensure your config has `data.flattened_data: False`.
    
    Algorithm: Uses 2x2 grid summing to preserve total force at each LOD level.
    Odd spatial dimensions are supported (final partial blocks are preserved).
    
    Args:
        pressure_data: (N, H, W, 2) array where:
                       N = number of samples
                       H, W = spatial dimensions (typically 60, 21 for foot pressure)
                       Last dimension (2) = [left_foot, right_foot]
        lod_level: int, 0-5 indicating desired LOD level
                  0 = full resolution (no downsampling)
                  1 = 1x downsampled (60x21 -> 30x11)
                  2-5 = progressively further downsampled
                  
    Returns:
        Downsampled pressure data: (N, H', W', 2) where H' and W' depend on LOD level
        
    Raises:
        ValueError: if lod_level not in range 0-5 or data format is incorrect
    """
    if lod_level < 0 or lod_level > 5:
        raise ValueError(f"LOD level must be 0-5, got {lod_level}")
    
    if lod_level == 0:
        return pressure_data
    
    # Check input format
    if len(pressure_data.shape) != 4:
        raise ValueError(
            f"Expected 4D pressure data (N, H, W, 2), got shape {pressure_data.shape}. "
            f"Ensure config has `data.flattened_data: False` to load spatial pressure data."
        )
    
    n_samples, h, w, n_feet = pressure_data.shape
    
    # Verify foot dimension
    if n_feet != 2:
        raise ValueError(f"Expected 2 feet (left/right), got {n_feet}")
    
    # Apply LOD downsampling progressively.
    # Treat NaNs as missing sensors that should not contribute to a
    # neighborhood sum: replace them with 0 for the pooling so that
    # valid values are preserved and NaNs do not poison the sum.
    current_data = np.nan_to_num(pressure_data.copy(), nan=0.0)
    for _ in range(lod_level):
        row_idx = np.arange(0, current_data.shape[1], 2)
        col_idx = np.arange(0, current_data.shape[2], 2)
        # Sum-pool in spatial dims only; channel dim (left/right foot) is kept.
        current_data = np.add.reduceat(
            np.add.reduceat(current_data, row_idx, axis=1),
            col_idx,
            axis=2,
        )
    
    return current_data

