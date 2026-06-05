import numpy as np
import cv2
from module.match import detect_and_match

def update_tracks(image1, image2, img_idx, current_tracks, kp_to_track, next_id):
    _, _, kp1, kp2, good_matches = detect_and_match(image1, image2)
    for m in good_matches:
        q_idx = m.queryIdx # Index in previous image
        t_idx = m.trainIdx # Index in current image
        
        # Check if the point in the previous image already has a track ID i
        if (img_idx - 1, q_idx) in kp_to_track:
            track_id = kp_to_track[(img_idx - 1, q_idx)]
        else:
            # Assign a new unique index i
            track_id = next_id
            next_id += 1
            
            # Record the starting point {xi, yi}
            current_tracks[track_id] = {img_idx - 1: kp1[q_idx].pt}
            kp_to_track[(img_idx - 1, q_idx)] = track_id
        
        # Add the current image's coordinates to the same track i
        current_tracks[track_id][img_idx] = kp2[t_idx].pt
        kp_to_track[(img_idx, t_idx)] = track_id
        
    return current_tracks, kp_to_track, next_id

def build_track(images):
    current_tracks = {}   # The database of {track_id: {img_idx: (x,y)}}
    kp_to_track = {}      # The memory map {(img_idx, kp_idx): track_id}
    next_id = 0           # The unique index 'i' counter

    for i in range(len(images) - 1):
        img_prev =images[i]
        img_curr =images[i+1]
        
        current_tracks, kp_to_track, next_id = update_tracks(
            img_prev, 
            img_curr,
            img_idx=i+1, 
            current_tracks=current_tracks, 
            kp_to_track=kp_to_track, 
            next_id=next_id
        )
        print(f"Frame {i} -> {i+1} processed. Total tracks: {next_id}")
    
    return current_tracks
