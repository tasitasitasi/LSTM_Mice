num_frames, num_mice, _, num_bodyparts = keypoints.shape
    
    #  key bodypart indices
    bp_dict = {bp: i for i, bp in enumerate(bodyparts)}
    
    #  bodyparts for behavior detection
    nose_idx = bp_dict.get('nose', 0)
    neck_idx = bp_dict.get('neck', bp_dict.get('back_of_neck', 1))
    tail_idx = bp_dict.get('tail', bp_dict.get('base_of_tail', -1))
    
    #  center point (average of all bodyparts) for each mouse
    # Shape: (num_frames, num_mice, 2)
    mouse_centers = np.mean(keypoints, axis=3)
    
    #  nose positions for each mouse
    # Shape: (num_frames, num_mice, 2)
    mouse_noses = keypoints[:, :, :, nose_idx]
    
    feature_list = []
    feature_names = []
    
    # raw keypoints
    raw_features = keypoints.reshape(num_frames, -1)
    feature_list.append(raw_features)
    for m in range(num_mice):
        for bp in bodyparts:
            feature_names.extend([f'mouse{m}_{bp}_x', f'mouse{m}_{bp}_y'])
    
    # intermouse distance 
    if num_mice >= 2:
        # Distance between mouse centers
        center_dist = np.sqrt(np.sum((mouse_centers[:, 0] - mouse_centers[:, 1])**2, axis=1))
        feature_list.append(center_dist.reshape(-1, 1))
        feature_names.append('inter_mouse_center_distance')
        
        # distance between noses
        nose_dist = np.sqrt(np.sum((mouse_noses[:, 0] - mouse_noses[:, 1])**2, axis=1))
        feature_list.append(nose_dist.reshape(-1, 1))
        feature_names.append('inter_mouse_nose_distance')
        
        # Nose-to-body distances (for sniffing)
        # Mouse 0's nose to Mouse 1's center
        nose0_to_body1 = np.sqrt(np.sum((mouse_noses[:, 0] - mouse_centers[:, 1])**2, axis=1))
        feature_list.append(nose0_to_body1.reshape(-1, 1))
        feature_names.append('mouse0_nose_to_mouse1_body')
        
        # Mouse 1's nose to Mouse 0's center
        nose1_to_body0 = np.sqrt(np.sum((mouse_noses[:, 1] - mouse_centers[:, 0])**2, axis=1))
        feature_list.append(nose1_to_body0.reshape(-1, 1))
        feature_names.append('mouse1_nose_to_mouse0_body')
    
    #how fast each mouse is moving 
    for m in range(num_mice):
        # Velocity = change in position between frames
        # Shape: (num_frames, 2)
        velocity = np.zeros((num_frames, 2))
        velocity[1:] = mouse_centers[1:, m] - mouse_centers[:-1, m]
        
        # smooth to reduce noise
        velocity[:, 0] = gaussian_filter1d(velocity[:, 0], sigma=smooth_sigma)
        velocity[:, 1] = gaussian_filter1d(velocity[:, 1], sigma=smooth_sigma)
        
        # Speed 
        speed = np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2)
        
        feature_list.append(velocity)
        feature_names.extend([f'mouse{m}_velocity_x', f'mouse{m}_velocity_y'])
        
        feature_list.append(speed.reshape(-1, 1))
        feature_names.append(f'mouse{m}_speed')
    
    # acceleration
    for m in range(num_mice):
        # Acceleration = change in velocity
        velocity = np.zeros((num_frames, 2))
        velocity[1:] = mouse_centers[1:, m] - mouse_centers[:-1, m]
        
        acceleration = np.zeros((num_frames, 2))
        acceleration[1:] = velocity[1:] - velocity[:-1]
        
        # Smooth
        acceleration[:, 0] = gaussian_filter1d(acceleration[:, 0], sigma=smooth_sigma)
        acceleration[:, 1] = gaussian_filter1d(acceleration[:, 1], sigma=smooth_sigma)
        
        accel_magnitude = np.sqrt(acceleration[:, 0]**2 + acceleration[:, 1]**2)
        
        feature_list.append(accel_magnitude.reshape(-1, 1))
        feature_names.append(f'mouse{m}_acceleration')
    
    # imporving approach
    if num_mice >= 2:
        # Rate of change of distance between mice
        # Negative = approaching, Positive = retreating
        distance_change = np.zeros(num_frames)
        distance_change[1:] = center_dist[1:] - center_dist[:-1]
        distance_change = gaussian_filter1d(distance_change, sigma=smooth_sigma)
        
        feature_list.append(distance_change.reshape(-1, 1))
        feature_names.append('approach_retreat_rate')
        
        # Smoothed version for longer-term trends
        distance_change_smooth = gaussian_filter1d(distance_change, sigma=smooth_sigma * 3)
        feature_list.append(distance_change_smooth.reshape(-1, 1))
        feature_names.append('approach_retreat_rate_smooth')
    
    # orientation features
    for m in range(num_mice):
        # Heading direction (nose relative to body center)
        heading = mouse_noses[:, m] - mouse_centers[:, m]
        heading_angle = np.arctan2(heading[:, 1], heading[:, 0])
        
        feature_list.append(np.sin(heading_angle).reshape(-1, 1))
        feature_list.append(np.cos(heading_angle).reshape(-1, 1))
        feature_names.extend([f'mouse{m}_heading_sin', f'mouse{m}_heading_cos'])
    
    # angle to see if mouse is facing another 
    if num_mice >= 2:
        # Vector from mouse 0 to mouse 1
        to_other = mouse_centers[:, 1] - mouse_centers[:, 0]
        angle_to_other = np.arctan2(to_other[:, 1], to_other[:, 0])
        
        # Mouse 0's heading
        heading0 = mouse_noses[:, 0] - mouse_centers[:, 0]
        heading0_angle = np.arctan2(heading0[:, 1], heading0[:, 0])
        
        # Angle difference (0 = facing directly at other mouse)
        facing_angle_0 = angle_to_other - heading0_angle
        # Normalize to [-pi, pi]
        facing_angle_0 = np.arctan2(np.sin(facing_angle_0), np.cos(facing_angle_0))
        
        feature_list.append(np.abs(facing_angle_0).reshape(-1, 1))
        feature_names.append('mouse0_facing_angle')
        
        # Same for mouse 1
        to_other_1 = mouse_centers[:, 0] - mouse_centers[:, 1]
        angle_to_other_1 = np.arctan2(to_other_1[:, 1], to_other_1[:, 0])
        
        heading1 = mouse_noses[:, 1] - mouse_centers[:, 1]
        heading1_angle = np.arctan2(heading1[:, 1], heading1[:, 0])
        
        facing_angle_1 = angle_to_other_1 - heading1_angle
        facing_angle_1 = np.arctan2(np.sin(facing_angle_1), np.cos(facing_angle_1))
        
        feature_list.append(np.abs(facing_angle_1).reshape(-1, 1))
        feature_names.append('mouse1_facing_angle')
    
    # posture indicators
    for m in range(num_mice):
        # Body length (nose to tail)
        if tail_idx >= 0:
            body_length = np.sqrt(np.sum(
                (keypoints[:, m, :, nose_idx] - keypoints[:, m, :, tail_idx])**2, 
                axis=1
            ))
            feature_list.append(body_length.reshape(-1, 1))
            feature_names.append(f'mouse{m}_body_length')
    
    # combine all features 
    features = np.hstack(feature_list)
    
    # Handle any NaN or Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features, feature_names


def normalize_features(features, method='standardize'):
  
   # Normalize features for better training.
 
    if method == 'standardize':
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized = (features - mean) / std
        stats = {'mean': mean, 'std': std, 'method': 'standardize'}
    else:  # minmax
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        normalized = (features - min_val) / range_val
        stats = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    
    return normalized, stats


def apply_normalization(features, stats):
    """Apply saved normalization to new data (e.g., test set)."""
    if stats['method'] == 'standardize':
        return (features - stats['mean']) / stats['std']
    else:
        range_val = stats['max'] - stats['min']
        range_val[range_val == 0] = 1
        return (features - stats['min']) / range_val


#test
if __name__ == "__main__":
    print("Testing feature engineering...")
    
    # Create dummy data: 100 frames, 2 mice, 2 coords (x,y), 7 bodyparts
    dummy_keypoints = np.random.randn(100, 2, 2, 7) * 100
    dummy_bodyparts = ['nose', 'left_ear', 'right_ear', 'neck', 'left_hip', 'right_hip', 'tail']
    
    features, feature_names = compute_features(dummy_keypoints, dummy_bodyparts)
    

    print(f"   Input shape: {dummy_keypoints.shape}")
    print(f"   Output shape: {features.shape}")
    print(f"   Number of features: {len(feature_names)}")
    print(f"\n   Feature categories:")
    print(f"   - Raw keypoints: {2 * 2 * 7} features")
    print(f"   - Distance features: 4 features")
    print(f"   - Velocity features: 6 features (2 mice × 3)")
    print(f"   - Acceleration: 2 features")
    print(f"   - Approach/retreat: 2 features")
    print(f"   - Heading: 4 features")
    print(f"   - Facing angle: 2 features")
    print(f"   - Body length: 2 features")
    print(f"\n   Sample feature names:")
    for name in feature_names[:5]:
        print(f"     - {name}")
    print("     ...")
    for name in feature_names[-5:]:
        print(f"     - {name}")
