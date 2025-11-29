import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


def load_parquet_tracking(file_path):
    """
    Load tracking data from parquet and convert to array format
    """
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

    # init array: (frames, mice, coords, bodyparts)
    keypoints = np.zeros((num_frames, num_mice, 2, num_bodyparts))

    for _, row in df.iterrows():
        f_idx = frame_map[row['video_frame']]
        m_idx = mouse_map[row['mouse_id']]
        b_idx = bodypart_map[row['bodypart']]
        keypoints[f_idx, m_idx, 0, b_idx] = row['x']
        keypoints[f_idx, m_idx, 1, b_idx] = row['y']

    return keypoints, bodyparts, frames


def convert_intervals_to_frames(annot_df, num_frames):

    # init all frames as "other" (for no specific behavior)
    frame_labels = ['other'] * num_frames

    # get vocabulary (all unique actions)
    vocabulary = sorted(annot_df['action'].unique())

    # fill in annotated behaviors
    for _, row in annot_df.iterrows():
        action = row['action']
        start = int(row['start_frame'])
        stop = int(row['stop_frame'])

        # label all frames in this interval
        for frame_idx in range(start, min(stop + 1, num_frames)):
            if 0 <= frame_idx < num_frames:
                # If multiple behaviors overlap, keep the first one
                if frame_labels[frame_idx] == 'other':
                    frame_labels[frame_idx] = action

    return frame_labels, vocabulary


def load_training_data(tracking_dir, annotation_dir, max_sequences=None):
    data = {
        'sequences': {},
        'vocabulary': set()
    }

    tracking_path = Path(tracking_dir)
    annotation_path = Path(annotation_dir)


    lab_folders = [f for f in tracking_path.iterdir() if f.is_dir()]
    print(f"Found {len(lab_folders)} lab folders")

    total_sequences = 0
    skipped_labs = []

    for lab_folder in lab_folders:
        lab_name = lab_folder.name

        # check
        annot_lab_folder = annotation_path / lab_name
        if not annot_lab_folder.exists():
            skipped_labs.append(lab_name)
            continue

        print(f"\n  ✓ Processing lab: {lab_name}")

        tracking_files = list(lab_folder.glob('*.parquet'))

        matched = 0
        for track_file in tracking_files:
            if max_sequences and total_sequences >= max_sequences:
                break

            seq_id = track_file.stem

            # load
            try:
                keypoints, bodyparts, frame_numbers = load_parquet_tracking(track_file)
                num_frames = keypoints.shape[0]
            except Exception as e:
                print(f"      Error loading {seq_id}: {e}")
                continue

            # load
            annot_file = annot_lab_folder / f"{seq_id}.parquet"

            if annot_file.exists():
                try:
                    annot_df = pd.read_parquet(annot_file)

                    # convert ints
                    frame_labels, vocab = convert_intervals_to_frames(annot_df, num_frames)

                    #  global vocabulary
                    data['vocabulary'].update(vocab)

                    data['sequences'][seq_id] = {
                        'keypoints': keypoints,
                        'annotations': frame_labels,
                        'annotator_id': 0,
                        'lab': lab_name,
                        'bodyparts': bodyparts
                    }

                    matched += 1
                    total_sequences += 1

                except Exception as e:
                    print(f"      Error with annotations for {seq_id}: {e}")

            if max_sequences and total_sequences >= max_sequences:
                break

        print(f"    Loaded {matched}/{len(tracking_files)} sequences")

        if max_sequences and total_sequences >= max_sequences:
            break

    # convert vocabulary to sorted list and add "other"
    vocab_list = sorted(list(data['vocabulary']))
    if 'other' not in vocab_list:
        vocab_list.insert(0, 'other')
    data['vocabulary'] = vocab_list

    if skipped_labs:
        print(f"\n⏭  Skipped {len(skipped_labs)} unlabeled datasets")

    print(f"\n Loaded {total_sequences} annotated sequences total")
    return data


# main execution
if __name__ == "__main__":
    tracking_dir = '/Users/tasi/Desktop/MABE/data/raw/train_tracking'
    annotation_dir = '/Users/tasi/Desktop/MABE/data/raw/train_annotation'

    print("=" * 60)
    print("LOADING TRAINING DATA (SAMPLE)")
    print("=" * 60)

    # Load a sample first 
    train_data = load_training_data(tracking_dir, annotation_dir, max_sequences=10)


    print(f"\n Total sequences loaded: {len(train_data['sequences'])}")
    print(f" Vocabulary ({len(train_data['vocabulary'])} classes): {train_data['vocabulary']}")

    if len(train_data['sequences']) > 0:
        # Check first sequence
        first_seq_id = list(train_data['sequences'].keys())[0]
        first_seq = train_data['sequences'][first_seq_id]

        print(f"\n Sample Sequence:")
        print(f"   ID: {first_seq_id}")
        print(f"   Lab: {first_seq.get('lab', 'Unknown')}")
        print(f"   Keypoints shape: {first_seq['keypoints'].shape}")
        print(f"   Bodyparts: {first_seq.get('bodyparts', [])}")
        print(f"   Annotations length: {len(first_seq['annotations'])} frames")

        # Show distribution of labels in this sequence
        label_counts = Counter(first_seq['annotations'])
        print(f"\n   Behavior distribution in this sequence:")
        for behavior, count in label_counts.most_common():
            pct = 100 * count / len(first_seq['annotations'])
            print(f"      {behavior:20s}: {count:5d} frames ({pct:5.1f}%)")

        # Analyze all sequences
        print(f"\n Overall Dataset Statistics:")
        all_labels = []
        sequence_lengths = []

        for seq_id, seq_data in train_data['sequences'].items():
            all_labels.extend(seq_data['annotations'])
            sequence_lengths.append(len(seq_data['annotations']))

        label_counts = Counter(all_labels)
        total_frames = len(all_labels)

        print(f"   Total frames: {total_frames:,}")
        print(f"   Average sequence length: {np.mean(sequence_lengths):.1f} frames")

        print(f"\n  Class Distribution (all sequences):")
        for behavior in train_data['vocabulary']:
            count = label_counts.get(behavior, 0)
            percentage = (count / total_frames) * 100 if total_frames > 0 else 0
            print(f"   {behavior:20s}: {count:7,} frames ({percentage:5.2f}%)")

        # plot
        plt.figure(figsize=(12, 6))
        behaviors = train_data['vocabulary']
        counts = [label_counts.get(b, 0) for b in behaviors]

        colors = plt.cm.Set3(range(len(behaviors)))
        bars = plt.bar(behaviors, counts, color=colors, alpha=0.8, edgecolor='black')

        plt.xlabel('Behavior', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Frames', fontsize=12, fontweight='bold')
        plt.title('Class Distribution in Training Data (Sample)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                percentage = (count / total_frames) * 100
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{percentage:.1f}%',
                         ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        output_dir = Path('/Users/tasi/Desktop/MABE/docs')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'class_distribution_sample.png', dpi=300, bbox_inches='tight')
        plt.show()
