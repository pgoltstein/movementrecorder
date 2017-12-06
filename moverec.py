#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to record webcam video when movement is detected

Created on Mon Dec 4, 2017
@author: pgoltstein
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# User settings
# video_resolution = (1280,1024) # (x,y)
video_resolution = (640,480) # (x,y)
buffer_size = 60
record_min_n_frames = 50
mov_det_size = (160,120) # (x,y)
motion_compare_past = 10
# save_location = "/data/moverecdata"
save_location = "C:/Users/goltstein/OneDrive/moverecs"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports
import cv2
import numpy as np
import sys, os, time, datetime
from sys import platform as _platform
import argparse

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arguments
parser = argparse.ArgumentParser( \
    description = \
        "Runs a webcam continuously, and records only snippets of video" + \
        " where movement is detected. " + \
        "(written by Pieter Goltstein - December 2017)")

parser.add_argument('-t','--threshold', type=int, default=3,
                    help= 'Threshold for motion detection (default=3)')
parser.add_argument('-v', '--verbose', action="store_true",
    help='Prints motion quantification in terminal (on/off default=off)')
parser.add_argument('-d', '--difference', action="store_true",
    help='Displays pixelwise difference on screen (on/off default=off)')
parser.add_argument('-q', '--quiet', action="store_true",
    help='Runs in quiet mode, no output nor display (on/off default=off)')
args = parser.parse_args()
movement_threshold = args.threshold * (mov_det_size[0]*mov_det_size[1])
print_motion = args.verbose
display_difference = args.difference
no_output = args.quiet

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Detect operating system
if "linux" in _platform.lower():
   OS = "linux" # linux
elif "darwin" in _platform.lower():
   OS = "macosx" # MAC OS X
elif "win" in _platform.lower():
   OS = "windows" # Windows

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_resolution[1])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize window and buffer
if not no_output:
    cv2.imshow('Preview',np.zeros( (video_resolution[0],
                                    video_resolution[1]), dtype=int ))
frame_buffer = np.zeros( (video_resolution[1], video_resolution[0], 3,
                    buffer_size), dtype=np.uint8 )
date_time = ["" for x in range(buffer_size)]
buffer_ix = 1
current_ix = 0
prev_ix = buffer_size-motion_compare_past

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Configure location of timestamp
font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL
bottomLeftCornerOfText = (10,video_resolution[1]-10)
fontScale              = 1
fontColor              = (0,0,255)
lineThickness          = 1

# Capture and show frame-by-frame
save_img = 0
saving = False
frame_counter = 0.0
print("Starting frame buffer...")
while True:

    # Read frame
    ret, frame = cap.read()
    frame_counter += 1

    # Write timestamp
    date_time[current_ix] = \
        datetime.datetime.now().strftime("%d %b %Y - %H:%M:%S")

    # Put image in buffer and get present buffered frame
    frame_buffer[:,:,:,current_ix] = frame

    # Convert to BG for movement detection
    frame_bg = cv2.resize( cv2.cvtColor( frame,
        cv2.COLOR_BGR2GRAY),mov_det_size).astype(np.float)
    prev_frame_bg = cv2.resize( cv2.cvtColor( frame_buffer[:,:,:,prev_ix],
        cv2.COLOR_BGR2GRAY),mov_det_size).astype(np.float)

    # Normalize by subtracting the mean
    frame_bg = frame_bg - frame_bg.mean()
    prev_frame_bg = prev_frame_bg - prev_frame_bg.mean()

    # Check whether there was movement
    mv_sum = np.sum(np.abs(prev_frame_bg-frame_bg))
    if print_motion and frame_counter>buffer_size:
        print("Motion parameter = {}".format( mv_sum \
            /(mov_det_size[0]*mov_det_size[1])))
    if mv_sum > movement_threshold and frame_counter>buffer_size:
        save_img = buffer_size+record_min_n_frames

    # Create file if saving just turned on
    if save_img>0 and saving is False and frame_counter>buffer_size:
        date_time_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(save_location,date_time_path)

        # Define the codec and create VideoWriter object
        if OS == "macosx":
            video_file_name = save_path+'.mov'
            print("Detected motion, creating file: {}".format(video_file_name))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_object = cv2.VideoWriter( video_file_name,fourcc, 30.0,
                (video_resolution[0],video_resolution[1]) )
        elif OS == "windows":
            video_file_name = save_path+'.avi'
            print("Detected motion, creating file: {}".format(video_file_name))
            fourcc = cv2.VideoWriter_fourcc(*'divx')
            video_object = cv2.VideoWriter( video_file_name,fourcc, 30.0,
                (video_resolution[0],video_resolution[1]) )
        elif OS == "linux":
            video_file_name = save_path+'.avi'
            print("Detected motion, creating file: {}".format(video_file_name))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_object = cv2.VideoWriter( video_file_name,fourcc, 30.0,
                (video_resolution[0],video_resolution[1]) )
        saving = True
        write_counter = 0

    # Save the correct buffer frame
    if save_img > 0 and saving == True:
        buf_frame = frame_buffer[:,:,:,buffer_ix]
        # Now need to copy to make the slice / transpose physically take effect
        buf_frame = buf_frame.copy()
        cv2.putText(buf_frame, date_time[buffer_ix], bottomLeftCornerOfText,
            font, fontScale, fontColor, lineThickness)
        video_object.write(buf_frame)
        write_counter += 1
        save_img -= 1

    # Close file if saving is finished
    if save_img == 0 and saving == True:
        video_object.release()
        print(" -> done, wrote {} frames".format(write_counter))
        saving = False

    # Show on screen
    if not no_output:
        if display_difference:
            diff_im = (np.abs(prev_frame_bg-frame_bg)).astype(np.uint8)
            cv2.putText(diff_im, date_time[current_ix], bottomLeftCornerOfText,
                font, fontScale, fontColor, lineThickness)
            cv2.imshow('Preview',diff_im)
        else:
            cv2.putText(frame, date_time[current_ix], bottomLeftCornerOfText,
                font, fontScale, fontColor, lineThickness)
            cv2.imshow('Preview',frame)

    # Update buffer pointers
    current_ix += 1
    buffer_ix += 1
    prev_ix += 1
    if buffer_ix == buffer_size:
        buffer_ix = 0
    if current_ix == buffer_size:
        current_ix = 0
    if prev_ix == buffer_size:
        prev_ix = 0

    # Quit if escape pressed
    try:
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
    except:
        pass

# Release webcam and close window
cv2.destroyAllWindows()
