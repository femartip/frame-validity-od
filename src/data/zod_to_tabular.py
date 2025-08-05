import sys
import pandas as pd
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import random
import os
import argparse
import logging
from tqdm import tqdm

from zod import ZodFrames
from zod import ZodSequences
import zod.constants as constants
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject
from zod.data_classes import LidarData
from zod.visualization.oxts_on_image import visualize_oxts_on_image
from zod import ObjectAnnotation
from zod.visualization.object_visualization import overlay_object_2d_box_on_image, overlay_object_3d_box_on_image
from zod.visualization.lidar_on_image import visualize_lidar_on_image
from zod import EgoRoadAnnotation
from zod.utils.polygon_transformations import polygons_to_binary_mask
from zod.utils.polygon_transformations import polygons_to_binary_mask
from zod.visualization.polygon_utils import overlay_mask_on_image
from zod import LaneAnnotation

import openmeteo_requests
import requests_cache
from retry_requests import retry
from tenacity import retry as tenacity_retry, wait_exponential

import cv2
import sys
from PIL import Image
from brisque import Brisque
from brisque import score as brisque_score
import torch
from torchmetrics import multimodal

DATA_DIR = "./data/metafeatures_1.csv"

def get_iqa(image: np.ndarray, func, *args) -> float:
    if isinstance(func, multimodal.CLIPImageQualityAssessment):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).unsqueeze(0) 

    blur = func(image, *args)

    if isinstance(func, multimodal.CLIPImageQualityAssessment):
        blur = blur.numpy()

    return blur

@tenacity_retry(wait=wait_exponential(multiplier=1, min=10, max=60))
def get_weather_from_api(coords: tuple[float, float], datatime_utc: pd.Timestamp) -> dict:
    """
    API day limit is 10000 requests
    Current implementation rounds the hour to the floor, this is not ideal. 
    """
    datatime_day = str(datatime_utc).split(" ")[0]

    logging.debug(f"Getting Weather for {coords} on the {datatime_day}")
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    variable_names = ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "sunshine_duration", "wind_speed_10m", "weather_code"]
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "start_date": datatime_day,
        "end_date": datatime_day,
        "hourly": ",".join(variable_names),
        "timezone": "UTC"
    }
    
    logging.debug(f"API Parameters: {params}")

    response = openmeteo.weather_api(url, params=params)[0]
    
    if not response or not response.Hourly():
        print("Invalid weather response received")
        return {}
    
    hourly = response.Hourly()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    for i, var_name in enumerate(variable_names):
        variable = hourly.Variables(i)
        if variable:
            hourly_data[var_name] = variable.ValuesAsNumpy()
        else:
            print(f"Variable {i} ({var_name}) is None")
            hourly_data[var_name] = None
    
    
    hourly_dataframe = pd.DataFrame(data = hourly_data)
    weather_dict = hourly_dataframe[hourly_dataframe['date'] == datatime_utc].iloc[0].drop('date').to_dict()    
    logging.debug(f"Weather informatio for instance {weather_dict}")
    
    return weather_dict
    

def get_data(zod_frames):
    training_frames = zod_frames.get_split(constants.TRAIN)
    validation_frames = zod_frames.get_split(constants.VAL)

    logging.info(f"Number of training frames: {len(training_frames)}") 
    logging.info(f"Number of validation frames: {len(validation_frames)}")
    return training_frames, validation_frames

def get_caracteristics(training_frames, zod_frames, num_frames: int, prev_frames_id: List[str]) -> tuple[dict, list]:
    total_frames_id = [zod_frames[frame_id].metadata.frame_id for frame_id in training_frames]
    frames_id = list(set(total_frames_id) - set(prev_frames_id))
    if len(frames_id) > num_frames:
        frames_id = frames_id[:num_frames]
    logging.info(f"Frames not in csv {len(frames_id)}")
    #print(total_frames_id)

    
    data_dict = {  
        "country": [], 
        "time_of_day": [], 
        "lat": [], 
        "long": [], 
        "road_type": [],
        "road_condition": [], 
        "weather": [],
        "solar_angle_elevation": [],
        "month": [],  
        "hour": [],
    }   

    for id in tqdm(frames_id):
        data_dict["country"].append(zod_frames[id].metadata.country_code)
        data_dict["time_of_day"].append(zod_frames[id].metadata.time_of_day)
        data_dict["lat"].append(zod_frames[id].metadata.latitude)
        data_dict["long"].append(zod_frames[id].metadata.longitude)
        data_dict["road_type"].append(zod_frames[id].metadata.road_type)
        data_dict["road_condition"].append(zod_frames[id].metadata.road_condition)
        data_dict["weather"].append(zod_frames[id].metadata.scraped_weather) 
        data_dict["solar_angle_elevation"].append(zod_frames[id].metadata.solar_angle_elevation)
        #data_dict["acc"].append(zod_frames[id].ego_motion.accelerations)
        #data_dict["ang_rates"].append(zod_frames[id].ego_motion.angular_rates)
        #data_dict["orig_lat_lon"].append(zod_frames[id].ego_motion.origin_lat_lon)
        #data_dict["poses"].append(zod_frames[id].ego_motion.poses)
        #data_dict["vel"].append(zod_frames[id].ego_motion.velocities)
        #data_dict["timestamp"].append(zod_frames[id].ego_motion.timestamps)
        #data_dict["calid_cam"].append(zod_frames[id].calibration.cameras)
        #data_dict["datetime"].append(str(zod_frames[id].info.keyframe_time))
        data_dict["month"].append(zod_frames[id].info.keyframe_time.month)
        data_dict["hour"].append(zod_frames[id].info.keyframe_time.hour)            #In UTC+0, not local time

        weather_dict = get_weather_from_api((zod_frames[id].metadata.latitude, zod_frames[id].metadata.longitude), pd.to_datetime(str(zod_frames[id].info.keyframe_time), utc=True).round('h'))
        data_dict.update(weather_dict)

        zod_frame = zod_frames[id]
        image = zod_frame.get_image()

        func = {"brisque":brisque_score, "quality":multimodal.CLIPImageQualityAssessment(), "brightness": multimodal.CLIPImageQualityAssessment(prompts=("brightness",)),"noisiness": multimodal.CLIPImageQualityAssessment(prompts=("noisiness",)), "sharpness": multimodal.CLIPImageQualityAssessment(prompts=("sharpness",)), "contrast": multimodal.CLIPImageQualityAssessment(prompts=("contrast",)), "complexity": multimodal.CLIPImageQualityAssessment(prompts=("complexity",))}
        iqa = {k: get_iqa(image ,v) for k, v in func.items()}
        data_dict.update(iqa)

    return data_dict, frames_id

def to_csv(data_dict: dict[str, list], frames_id: list, resume: bool) -> None:
    base_dir = "/".join(DATA_DIR.split("/")[:-1])
    assert os.path.exists(base_dir), f"File {base_dir} does not exist, cant save csv"
    if resume:
        prev_df = pd.read_csv(DATA_DIR, index_col=0)
        current_df = pd.DataFrame(data=data_dict, index= frames_id)
        df = pd.concat([prev_df, current_df])
        df.to_csv(DATA_DIR)
        print(f"Data saved to {DATA_DIR}, length of dataframe {df.shape[0]}")
    else:
        df = pd.DataFrame(data=data_dict, index= frames_id)
        df.to_csv(DATA_DIR, index=True)
        print("New csv saved")

def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("num_entries", type=int)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = argparser()
    dataset_root = "./data/zod"  
    version = "full"  
    zod_frames = ZodFrames(dataset_root=dataset_root, version=version)

    train, val = get_data(zod_frames)

    prev_frames_id = []
    if args.resume == True and os.path.exists(DATA_DIR):
        df = pd.read_csv(DATA_DIR, index_col=0)
        prev_frames_id.extend(df.index.values.tolist())
        prev_frames_id = [f"{int(item):06d}" for item in prev_frames_id]

    logging.debug(f"List of previous ids: {prev_frames_id}")
    #data_dict, id = get_caracteristics(train, zod_frames, args.num_entries, prev_frames_id)
    data_dict, id = get_caracteristics(val, zod_frames, args.num_entries, prev_frames_id)
    to_csv(data_dict, id, args.resume)