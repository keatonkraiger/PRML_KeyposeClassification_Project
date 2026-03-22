from re import A, I
from imutils.video import VideoStream
from imutils.video import FPS

import argparse
import cv2
from glob import glob
import json
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import h5py

def crop_video(video_path, mocap_file, crop_size, save_dir, annot_df, subject_id, take_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cropped_video = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, (crop_size, crop_size))
    
    # Load Mocap and check validity
    mocap_available = False
    try:
        with h5py.File(mocap_file, 'r') as f:
            joints = np.array(f['POSE']) # Expect shape [3, joints, frames]
            mocap_available = True
    except:
        print(f"[WARN] No Mocap for Sub {subject_id} Take {take_id}")

    # Get Manual Label for this take
    manual_row = annot_df[(annot_df['subject'] == int(subject_id)) & (annot_df['take'] == int(take_id))]
    manual_x = manual_row.iloc[0]['hip_x'] if not manual_row.empty else None
    manual_y = manual_row.iloc[0]['hip_y'] if not manual_row.empty else None
    manual_frame = int(manual_row.iloc[0]['frame_index']) if not manual_row.empty else -1

    # Calculate offset ONLY if Mocap is actually detecting at the keyframe
    offset_x, offset_y = 0, 0
    if mocap_available and manual_frame != -1:
        # Check third dimension (index 2) for detection flag
        if joints[2, 12, manual_frame] > 0: 
            offset_x = manual_x - joints[0, 12, manual_frame]
            offset_y = manual_y - joints[1, 12, manual_frame]

    # Initialize previous positions to center of screen as absolute fallback
    prev_x, prev_y = width // 2, height // 2
    frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        target_x, target_y = None, None

        # 1. Try Mocap (Check detection dim)
        if mocap_available and frames < joints.shape[2]:
            if joints[2, 12, frames] > 0: # PELVIS DETECTED
                target_x = joints[0, 12, frames] + offset_x
                target_y = joints[1, 12, frames] + offset_y

        # 2. Try Manual Fallback if Mocap failed/is missing
        if target_x is None and manual_x is not None:
            target_x, target_y = manual_x, manual_y

        # 3. Last Resort: Stay where we were
        if target_x is None:
            breakpoint()
            target_x, target_y = prev_x, prev_y

        # Smoothing & Boundary Logic
        target_x, target_y = int(target_x), int(target_y)
        
        # Crop math with padding
        x1 = target_x - (crop_size // 2)
        y1 = target_y - (crop_size // 2)
        
        # Apply padding if crop is outside frame boundaries
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, (x1 + crop_size) - width)
        pad_b = max(0, (y1 + crop_size) - height)

        if any([pad_l, pad_t, pad_r, pad_b]):
            frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
            x1 += pad_l
            y1 += pad_t

        crop = frame[y1 : y1 + crop_size, x1 : x1 + crop_size]
        
        # Final safety resize (VideoWriter is picky about size)
        if crop.shape[0:2] != (crop_size, crop_size):
            crop = cv2.resize(crop, (crop_size, crop_size))

        cropped_video.write(crop)
        prev_x, prev_y = target_x, target_y
        frames += 1

    cap.release()
    cropped_video.release()
    
  
def splice_video(subject, subject_dir, keyframe_labels, sub_take_path, M, N, save_dir, view='V1'):
    """ Given a subject, splice the videos into M seconds before and N seconds after each keyframe 
    for each take.

    Args
        subject: current subject 
        video_path: path to video file
        keyframe_labels: path to keyframe label file
        save_dir: path to save spliced videos
    """
    # Assumes directory naming convention of SubjectX - x=1:10
    subject_number = int(subject.partition("Subject")[-1])
    all_keyframes = np.loadtxt(keyframe_labels, delimiter=',')
    sub_takes = pd.read_csv(sub_take_path)
    takes = sub_takes[sub_takes['subject'] == subject_number]['take'].values

    videos_created = 0

    for take in takes:
        save_dir_base = os.path.join(save_dir, f"Take_{str(take)}")
        if not os.path.exists(save_dir_base):
            os.makedirs(save_dir_base)

        video_path = os.path.join(subject_dir, f"Video_{view}_{take}.mp4")
        col = sub_takes.query(f'subject=={subject_number} & take=={take}').index[0]
        keyframes = all_keyframes[:, col]
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Precompute start and end frames for each keyframe
        start_end_frames = []
        for keyframe in keyframes:
            if np.isnan(keyframe):
                continue
            start_frame = int(keyframe - M * fps)
            end_frame = int(keyframe + N * fps)
            if end_frame > total_frames:
                end_frame = total_frames
            if start_frame < 0:
                start_frame = 0
            start_end_frames.append((start_frame, end_frame))

        current_keyframe_idx = 0
        current_frame_idx = 0
        take_keypose_video = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_frame, end_frame = start_end_frames[current_keyframe_idx]
            if current_frame_idx >= start_frame and current_frame_idx <= end_frame:
                if take_keypose_video is None:
                    take_keypose_video = cv2.VideoWriter(os.path.join(save_dir_base, f"keypose_{current_keyframe_idx}.mp4"), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

                take_keypose_video.write(frame)

            if current_frame_idx == end_frame:
                take_keypose_video.release()
                take_keypose_video = None
                videos_created += 1
                current_keyframe_idx += 1
                if current_keyframe_idx == len(start_end_frames):
                    break

            current_frame_idx += 1

        cap.release()

def generate_label_file(subjects, class_names, save_path):
    """ Generate a label file for all videos in the dataset.

    Args
        subjects: list of subjects
        class_names: list of class names
        save_path: path to save label file
    """
    annot_paths = []
    annot_labels = []
    classes = list(class_names.keys())
    for subject in subjects:
        # Go through each take directory and get all video paths
        for take_dir in os.listdir(subject):
            if not os.path.isdir(os.path.join(subject, take_dir)):
                continue
            for video_path in os.scandir(os.path.join(subject, take_dir)):
                if not video_path.name.endswith(".mp4"):
                    continue
                class_num = video_path.name.split(".")[0]
                class_num = int(class_num.split("_")[-1])
                annot_paths.append(video_path.path)
                annot_labels.append(class_num)

    # Split labels into train and val sets. Randomly select 90% of the data for training and 10% for validation
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []

    val_inds = np.random.choice(len(annot_paths), int(len(annot_paths) * 0.1), replace=False)
    train_inds = np.setdiff1d(np.arange(len(annot_paths)), val_inds)
    for i in train_inds:
        train_paths.append(annot_paths[i])
        train_labels.append(annot_labels[i])
    for i in val_inds:
        val_paths.append(annot_paths[i])
        val_labels.append(annot_labels[i])

    # Save to file
    with open(save_path, "w") as f:
        for path, label in zip(train_paths,train_labels):
            f.write(f"{path}|{label}\n")

    if len(subjects) > 1:
        with open(save_path.replace("train", "val"), 'w') as f:
            for path, label in zip(val_paths, val_labels):
                f.write(f"{path}|{label}\n")
    

def create_labels(spliced_vids, test_subs, class_names, save_dir):
    """ Creates labels for the training and testing set where each file contains the path to the video 
    and the class of the video: /path/to/video,class

    Args
        spliced_vids: path to spliced videos
        test_subs: subjects to test on during LOSO
        class_names: path to class names
        save_dir: path to save labels
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(spliced_vids):
        raise ValueError("Path to spliced videos does not exist")

    spliced_video_dirs = [path for path in glob(f'{spliced_vids}/Subject*/')]

    for sub in test_subs:
        test_sub_dir = ''
        train_sub_dirs = []
        for dir in spliced_video_dirs:
            if dir.find(sub) != -1:
                test_sub_dir = dir
            else:
                train_sub_dirs.append(dir)

        if test_sub_dir == '':
            breakpoint()
            raise ValueError("Test subject not found")
        if len(train_sub_dirs) == 0:
            raise ValueError("No training subjects found")

        # First generate labels for training set
        test_file = os.path.join(save_dir, sub + '_test.csv')
        generate_label_file([test_sub_dir], class_names, test_file)

        # Then generate labels for testing set
        train_file = os.path.join(save_dir, sub + '_train.csv')
        generate_label_file(train_sub_dirs, class_names, train_file)

def main():
    ap = argparse.ArgumentParser()
    # Crop arguments
    ap.add_argument("--psutmm_root", type=str, required=True, help="path to input video directory")
    ap.add_argument("--crop_videos", action='store_true', help="whether to crop videos")
    ap.add_argument("--crop_size", type=int, default=500, help="Size of the cropped image")
    ap.add_argument("--crop_save_dir", type=str, default="Data/cropped", help="Directory to save cropped videos")
    ap.add_argument('--views', nargs='+', type=str, default=['V1'], help='List of views to process')
    ap.add_argument('--subjects', nargs='+', type=int, default=[1,2,3,4,5,6,7,8,9,10], help='List of subjects to process')

    # Video splicing arguments
    ap.add_argument("--video_splice", action='store_true', help="whether to splice videos")
    ap.add_argument("--keyframe_labels", type=str, default='frame_labels/taiji_keyframes.csv', help="path to keyframe labels")
    ap.add_argument("--video_splice_save_dir", type=str, default="PSUTMM_Action/videos", help="Directory to save the action videos")
    ap.add_argument("--sub_take_path", type=str, default="frame_labels/sub_takes.csv", help="Path to csv file containing subject and take number")
    ap.add_argument("--m", type=int, default=2, help="Number of seconds included before keyframe")
    ap.add_argument("--n", type=int, default=0.4, help="Number of seconds included after keyframe")

    # Create training and testing labels
    ap.add_argument("--create_labels", action='store_true', help="whether to create training and testing labels")
    ap.add_argument("--class_names", type=str, default="taiji_classnames.json", help="path to class names")
    ap.add_argument('--loso_subs', type=list, default=['Subject1'], help='List of subjects to be used for loso')
    ap.add_argument("--label_save_dir", type=str, default="PSUTMM_Action/labels", help="Directory to save labels")

    # add two save dirs for joint and foot pressure
    ap.add_argument("--overwrite", action='store_true', help="whether to overwrite existing files")
    args = ap.parse_args()

    annot_df = pd.read_csv('assets/pelvis_annotations.csv')

    video_root = os.path.join(args.psutmm_root, 'Modality_wise', 'Video')
    mocap_root = os.path.join(args.psutmm_root, 'Subject_wise')
   
    # For each subject in video directory, crop each video. 
    if args.crop_videos:
        print("[INFO] Cropping videos")
        if not os.path.exists(args.crop_save_dir):
            os.makedirs(args.crop_save_dir)

        for subject in tqdm(os.listdir(video_root)):
            subject_id = subject.replace("Subject", "")
            if int(subject_id) not in args.subjects:
                continue
            
            subject_dir = os.path.join(video_root, subject)
            if not os.path.isdir(subject_dir):
                continue
            if not os.path.exists(subject_dir):
                os.makedirs(subject_dir)
            
            crop_save_dir = os.path.join(args.crop_save_dir, subject)
            if not os.path.exists(crop_save_dir):
                os.makedirs(crop_save_dir)

            for view in args.views:
                for video in glob(os.path.join(subject_dir, f"Video_{view}*.mp4")):
                    print("[INFO] processing {}".format(video))
                    take_id = os.path.basename(video).split('_')[-1].replace('.mp4', '')
                    
                    save_dir = os.path.join(crop_save_dir, os.path.basename(video))
                    if os.path.exists(save_dir) and not args.overwrite:
                        print("[INFO] {} already exists. Skipping...".format(save_dir))
                        continue
                    mocap_file = os.path.join(mocap_root, subject, os.path.basename(video).replace(f"Video_{view}", f"Mocap_{view}").replace('mp4', 'mat'))
                    crop_video(video, mocap_file, args.crop_size, save_dir, annot_df, subject_id, take_id)
                    
    # For each subject in video directory, splice each video.
    if args.video_splice:

        print("[INFO] Splicing videos")
        splice_save_path = os.path.join(args.video_splice_save_dir)
        if not os.path.exists(splice_save_path):
            os.makedirs(splice_save_path)

        for subject in tqdm(os.listdir(args.crop_save_dir)):
            subject_id = subject.replace("Subject", "")
            if int(subject_id) not in args.subjects:
                continue
            
            subject_dir = os.path.join(args.crop_save_dir, subject)
            if not os.path.isdir(subject_dir):
                continue

            print(f"[INFO] Splicing videos for {subject}")
            save_dir=os.path.join(splice_save_path, subject)
            for view in args.views:
                splice_video(subject, subject_dir, args.keyframe_labels, args.sub_take_path, args.m, args.n, save_dir, view=view)
            
    # Create training and testing labels
    if args.create_labels:
        print("[INFO] Creating training and testing labels")
        f = open(args.class_names, 'r')
        class_names = json.load(f)
        f.close()
        label_save_dir = os.path.join(args.label_save_dir)
        create_labels(args.video_splice_save_dir, args.loso_subs, class_names, label_save_dir)



if __name__ == "__main__":
    main()