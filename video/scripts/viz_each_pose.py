import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import progressbar
import cv2
import os
from random import randrange

sub_take_path = "frame_labels/sub_takes.csv"
keyframe_labels = "frame_labels/taiji_keyframes.csv"
subject = "Subject1"
subject_dir = "cropped_videos/Subject1"
M = 2
N = 0.4
save_dir = "data/Subject1/key_frames"



def splice_video(subject, subject_dir, keyframe_labels, sub_take_path, M, N, save_dir):
    """
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
    takes = sub_takes[sub_takes['subject'] == subject_number]
    takes=takes['take'].values

    bar = progressbar.ProgressBar(maxval=(len(takes)+1)*45, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    videos_created = 0

    # For each take, we create a video for each of the labeled keyframes
    for take in takes:
        save_dir_base = os.path.join(save_dir,f"Take_{str(take)}")
        if not os.path.exists(save_dir_base):
            os.makedirs(save_dir_base)

        video_path = os.path.join(subject_dir, f"Video_V1_{take}.mp4")
        col = sub_takes.query(f'subject=={subject_number} & take=={take}').index[0]
        keyframes=all_keyframes[:,col]
        cap = cv2.VideoCapture(video_path)
        width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frames = []

        # Read in all frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        frames = np.array(frames)

        # For each keypose, splice video and save

        # add 0th class (inbetween randomly selected keyframes)
        valid_rand_frame = False
        while not valid_rand_frame:
            rand_keyframe = randrange(0, len(keyframes)-1)
            if not math.isnan(keyframes[rand_keyframe]) and not math.isnan(keyframes[rand_keyframe+1]):
                valid_rand_frame = True

        rand_range = keyframes[rand_keyframe+1] - keyframes[rand_keyframe]
        rand_frame = int(keyframes[rand_keyframe+1] - (rand_range // 2))
        keyframes = np.insert(keyframes, 0, rand_frame, axis=0)
        pose = 0
        breakpoint()
        for keyframe in keyframes:
            if math.isnan(keyframe):
                continue
            #take_keypose_video = cv2.VideoWriter(os.path.join(save_dir_base, f"keypose_{pose}.mp4"), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (int(width), int(height)))

            # M seconds before and N seconds after keyframe
            start_frame = int(keyframe - int(M*fps))
            end_frame = int(keyframe + int(N*fps))

            # Check for out of bounds frames
            if end_frame > total_frames:
                end_frame = int(total_frames)
            if start_frame < 0:
                start_frame = 0
            # foot_presure = fp[start_frame:end_frame, :, :, :]
            # openpose_joints = joints[start_frame:end_frame, :, :]
            spliced_video = frames[start_frame:end_frame, :, :, :]
            # Save the middle frame of the spliced video
            cv2.imwrite(os.path.join(save_dir_base, f"keypose_{pose}.jpg"), spliced_video[int(len(spliced_video)/2)])
            #for frame in spliced_video:
                #take_keypose_video.write(frame)
            #take_keypose_video.release()
            videos_created += 1
            pose+=1
            bar.update(videos_created)
            breakpoint()
    bar.finish()

splice_video(subject, subject_dir, keyframe_labels, sub_take_path, M, N, save_dir)