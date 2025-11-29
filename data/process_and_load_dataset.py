# feature engineering module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src/features'))
from feature_engineering import compute_features, normalize_features


def load_parquet_tracking(file_path):
    df = pd.read_parquet(file_path)
    frames = sorted(df['video_frame'].unique())
    mice = sorted(df['mouse_id'].unique())
    bodyparts = sorted(df['bodypart'].unique())

    num_frames = len(frames)
    num_mice = len(mice)
    num_bodyparts = len(bodyparts)

    frame_map = {f: i for i, f in enumerate(frames)}
    mouse_map = {m: i for i, m in enumerate(mice)}
    bodypart_map = {b: i for i, b in enumerate(bodyparts)}

    keypoints = np.zeros((num_frames, num_mice, 2, num_bodyparts))

    for _, row in df.iterrows():
        f_idx = frame_map[row['video_frame']]
        m_idx = mouse_map[row['mouse_id']]
        b_idx = bodypart_map[row['bodypart']]
        keypoints[f_idx, m_idx, 0, b_idx] = row['x']
        keypoints[f_idx, m_idx, 1, b_idx] = row['y']

    return keypoints, bodyparts, frames


def convert_intervals_to_frames(annot_df, num_frames):
    """Convert interval annotations to frame-level labels."""
    frame_labels = ['other'] * num_frames
    vocabulary = sorted(annot_df['action'].unique())

    for _, row in annot_df.iterrows():
        action = row['action']
        start = int(row['start_frame'])
        stop = int(row['stop_frame'])

        for frame_idx in range(start, min(stop + 1, num_frames)):
            if 0 <= frame_idx < num_frames:
                if frame_labels[frame_idx] == 'other':
                    frame_labels[frame_idx] = action

    return frame_labels, vocabulary


######updating paths
 

  # quick check
    print(f"   Tracking dir exists: {tracking_dir.exists()}")
    print(f"   Annotation dir exists: {annotation_dir.exists()}")

    if not tracking_dir.exists() or not annotation_dir.exists():
        print("Directory not found!")
        exit(1)

#load all sequences 
    data = {
        'sequences': {},
        'vocabulary': set()
    }

    tracking_files = list(tracking_dir.glob('*.parquet'))
    print(f"\nFound {len(tracking_files)} tracking files")

    success_count = 0
    error_count = 0
    all_feature_names = None

    for idx, track_file in enumerate(tracking_files):
        if idx % 20 == 0:
            print(f"  Processing {idx}/{len(tracking_files)}...")

        seq_id = track_file.stem

        try:
            # Load raw keypoints
            keypoints, bodyparts, frames = load_parquet_tracking(track_file)
            num_frames = keypoints.shape[0]

            # Load annotations
            annot_file = annotation_dir / f"{seq_id}.parquet"
            if not annot_file.exists():
                error_count += 1
                continue

            annot_df = pd.read_parquet(annot_file)
            frame_labels, vocab = convert_intervals_to_frames(annot_df, num_frames)

            #engineered features computing
            features, feature_names = compute_features(keypoints, bodyparts)
            
            if all_feature_names is None:
                all_feature_names = feature_names
                print(f"\nComputed {len(feature_names)} features per frame")

            # Store both raw keypoints + engineered features
            data['sequences'][seq_id] = {
                'keypoints': keypoints,           # Keep raw for reference
                'features': features,             # NEW: engineered features
                'annotations': frame_labels,
                'annotator_id': 0,
                'lab': 'CalMS21_task1',
                'bodyparts': bodyparts,
                'feature_names': feature_names    # NEW: for reference
            }

            data['vocabulary'].update(vocab)
            success_count += 1

        except Exception as e:
            error_count += 1
            if idx < 5:
                print(f"    ❌ Error: {e}")



# merging sniff classes 
    behavior_mapping = {
        'sniff': 'sniff_any',
        'sniffbody': 'sniff_any',
        'sniffface': 'sniff_any',
        'sniffgenital': 'sniff_any',
    }



    for seq_data in data['sequences'].values():
        annotations = seq_data['annotations']
        new_annotations = []
        for label in annotations:
            if label in behavior_mapping:
                new_annotations.append(behavior_mapping[label])
            else:
                new_annotations.append(label)
        seq_data['annotations'] = new_annotations

    # building vocab
    all_labels = []
    for seq_data in data['sequences'].values():
        all_labels.extend(seq_data['annotations'])

    label_counts = Counter(all_labels)
    total_frames = len(all_labels)

    MIN_FRAMES = 1000
    behaviors_to_keep = []
    for behavior, count in label_counts.items():
        if count >= MIN_FRAMES:
            behaviors_to_keep.append(behavior)

    if 'other' in behaviors_to_keep:
        behaviors_to_keep.remove('other')
    behaviors_to_keep = ['other'] + sorted(behaviors_to_keep)

    print(f"\n📋 Final vocabulary ({len(behaviors_to_keep)} classes): {behaviors_to_keep}")

    # relabel rare behaviors
    for seq_data in data['sequences'].values():
        annotations = seq_data['annotations']
        new_annotations = []
        for label in annotations:
            if label in behaviors_to_keep:
                new_annotations.append(label)
            else:
                new_annotations.append('other')
        seq_data['annotations'] = new_annotations

    data['vocabulary'] = behaviors_to_keep
    data['feature_names'] = all_feature_names

    # global normalization stats 
    
    # Collect all features for normalization
    all_features = []
    for seq_data in data['sequences'].values():
        all_features.append(seq_data['features'])
    
    all_features = np.vstack(all_features)
    
    # Compute stats
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    feature_std[feature_std == 0] = 1  # Avoid division by zero
    
    data['normalization'] = {
        'mean': feature_mean,
        'std': feature_std,
        'method': 'standardize'
    }
    
    # apply normalization to all sequences
    print(f"   Applying normalization to all sequences...")
    for seq_data in data['sequences'].values():
        seq_data['features'] = (seq_data['features'] - feature_mean) / feature_std
#display

    all_labels_final = []
    for seq_data in data['sequences'].values():
        all_labels_final.extend(seq_data['annotations'])

    label_counts_final = Counter(all_labels_final)
    total_final = len(all_labels_final)

    for behavior in behaviors_to_keep:
        count = label_counts_final.get(behavior, 0)
        pct = 100 * count / total_final if total_final > 0 else 0
        print(f"   {behavior:20s}: {count:8,} frames ({pct:5.2f}%)")

# save 
    output_path.parent.mkdir(exist_ok=True, parents=True)

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

  
    print(f"   Total features per frame: {len(all_feature_names)}")
    print(f"   Sample feature names:")
    for name in all_feature_names[:10]:
        print(f"     - {name}")
    print(f"     ... and {len(all_feature_names) - 10} more")
  


if __name__ == "__main__":
    main()
