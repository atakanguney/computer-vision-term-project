import os
import subprocess

import pandas as pd
import numpy as np
import cv2
from pytube import YouTube


def read_event_data(event_data_path):
    event_data = pd.read_csv(event_data_path, header=None)
    columns = [
        "#YoutubeId",
        "VideoWidth",
        "VideoHeight",
        "ClipStartTime",
        "ClipEndTime",
        "EventStartTime",
        "EventEndTime",
        "EventStartBallX",
        "EventStartBallY",
        "EventLabel",
        "TrainValOrTest",
    ]

    event_data.columns = columns

    return event_data


def read_bboxes_data(bboxes_data_path):
    bboxes_data = pd.read_csv(bboxes_data_path, header=None)
    columns = [
        "Youtube ID",
        "Time",
        "Top-left x",
        "Top-left y",
        "Width",
        "Height",
        "player-id",
    ]

    bboxes_data.columns = columns

    return bboxes_data


def download_video(video_id):
    youtube_base = "http://youtube.com/watch?v="
    yt = YouTube(youtube_base + video_id)
    stream = yt.streams.first().download()
    stream.download()

    return stream.codecs


def extract_clips_ffmpeg(video_path, starts, ends):
    """Extracts clips from video between starts-ends
    Parameters
    ==========
    video_path: str
        Video to be extracted clips
    starts: list
        Start times of clips
    ends: list
        End times of clips
    """

    video_format = video_path.split(".")[1]

    for i, (start, end) in enumerate(zip(starts, ends)):
        cmd = [
            "/usr/local/bin/ffmpeg",
            "-i", video_path,
            "-ss", str(start / 1000),
            "-to", str(end / 1000),
            "-q:v", "10",
            "-r", "5",  # paper says frame rate is 6 but according to bounding boxes data it is 5
            "-vcodec", "vp8",
            "-acodec", "libvorbis",
            "-y", "clips/clip-{}.{}".format(i, video_format),
        ]

        subprocess.run(cmd)


def extract_events_ffmpeg(video_path, event_ends):
    """Extract events from the video
    Parameters
    ==========

    video_path: str
        Path of the video to be extracted events
    event_ends: list
        List of timestamp of event ends
    """

    if not os.path.exists("events"):
        os.mkdir("events")

    video_format = video_path.split(".")[1]

    for i, end in enumerate(event_ends):
        start = end - 4e3
        cmd = [
            "/usr/local/bin/ffmpeg",
            "-i", video_path,
            "-ss", str(start / 1e3),
            "-to", str(end / 1e3),
            "-q:v", "10",
            "-r", "6",
            "-vcodec", "vp8",
            "-acodec", "libvorbis",
            "-y", "events/event_{}.{}".format(i, video_format)
        ]

        print(" ".join(cmd))
        subprocess.run(cmd, stderr=subprocess.PIPE)


def extract_frames_from_event(event_path):
    """Yields frames from given event

    Parameters
    ==========
    event_path: str
        Path of the event
    """
    cap = cv2.VideoCapture(event_path)
    cur_timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)
    success, frame = cap.read()

    while success:
        yield frame, cur_timestamp
        success, frame = cap.read()
        cur_timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)

    cap.release()


def extract_players_from_frame(frame, bounding_boxes, width, height):
    """Extract Players Regions

    Parameters
    ==========
    frame: np.ndarray
        Frame
    bounding_boxes: list
        List of tuples (top_left_x, top_left_y, width, height)
    """

    players_regions = []

    bounding_boxes = [(int(width * x), int(height * y), int(width * w),
                       int(height * h)) for x, y, w, h in bounding_boxes]

    for x, y, w, h in bounding_boxes:
        players_regions.append(frame[y:y+h, x:x+w])

    return players_regions


def find_bboxestime_of_frame(relative_timestamp, end_timestamp, event_bboxtimes_map):
    bboxtimes_of_event = event_bboxtimes_map[end_timestamp]

    start_timestamp = end_timestamp - 4e6
    absolute_timestamp = relative_timestamp + start_timestamp

    idx = (np.abs(bboxtimes_of_event - absolute_timestamp)).argmin()

    return bboxtimes_of_event[idx]


def extract_players_from_frames(event_path, event_bboxtimes_map, end_timestamp, bboxes_data, width, height):
    for frame, relative_timestamp in extract_frames_from_event(event_path):
        bboxestime = find_bboxestime_of_frame(
            relative_timestamp, end_timestamp, event_bboxtimes_map)
        bboxes = bboxes_data.loc[bboxes_data.Time == bboxestime, [
            "Top-left x", "Top-left y", "Width", "Height", "player-id"]].values

        player_regions = extract_players_from_frame(
            frame, bboxes[:, :4], width, height)
        player_dict = dict(zip(bboxes[:, 4], player_regions))

        yield frame, player_dict


def create_events_map(event_data, events_paths_list):
    # Event path list is assumed to be sorted in event end time
    events_endtime = event_data.sort_values(
        by="EventEndTime").EventEndTime.values
    events_map = dict(zip((events_endtime * 1000).astype(np.int_), events_paths_list))

    return events_map


def create_eventbboxtimes_map(event_data, events_map):
    event_bboxtimes_map = {end: event_data.loc[(end - 4e6 <= event_data.Time) & (
        event_data.Time <= end), "Time"].values for end in events_map}

    return event_bboxtimes_map
