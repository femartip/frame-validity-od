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
from tenacity import retry as tenacity_retry, wait_exponential, stop_after_delay, RetryError

from math import atan2, degrees

import cv2
import sys
from PIL import Image
from brisque import Brisque
from brisque import score as brisque_score
import torch
from torchmetrics import multimodal

DATA_DIR = "./data/metafeatures2.csv"
WEATHER_API_TIMEOUT_SECONDS = 300  

logger = logging.getLogger(__name__)


def get_iqa(image: np.ndarray, func) -> dict[str, float] | float:
    """Compute image quality assessment using the provided function.

    Adds debug logs for input shape, function type, and resulting values.
    """
    logger.debug(
        "Starting get_iqa: shape=%s, func=%s",
        getattr(image, "shape", None), getattr(func, "__name__", type(func).__name__),
    )
    scale_percent = 50 
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim)
    args = []

    if isinstance(func, multimodal.CLIPImageQualityAssessment):
        image = torch.from_numpy(image)             #type: ignore
        image = image.permute(2, 0, 1).unsqueeze(0)             #type: ignore
    if func is cv2.Laplacian:
        args.append(cv2.CV_64F)

    logger.debug("Invoking IQA function: %s with args=%s", getattr(func, "__name__", type(func).__name__), args)
    blur = func(image, *args)

    if isinstance(func, multimodal.CLIPImageQualityAssessment):
        if isinstance(blur, dict):
            blur = {k:v.numpy() for k,v in blur.items()}
        else:
            blur = blur.numpy()
    elif func is cv2.Laplacian:
        blur = blur.var()
    else:
        blur = blur
    logger.debug("Finished get_iqa: result_type=%s", type(blur).__name__)
    return blur

@tenacity_retry(
    wait=wait_exponential(multiplier=1, min=10, max=60),
    stop=stop_after_delay(WEATHER_API_TIMEOUT_SECONDS),
    reraise=True,
)
def get_weather_from_api(coords: tuple[float, float], datatime_utc: pd.Timestamp) -> dict:
    """
    API day limit is 10000 requests
    Current implementation rounds the hour to the floor, this is not ideal. 
    """
    datatime_day = str(datatime_utc).split(" ")[0]

    logger.debug("Getting weather for coords=%s date=%s", coords, datatime_day)
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)          #type: ignore

    variable_names = ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "wind_speed_10m", "weather_code"]
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "start_date": datatime_day,
        "end_date": datatime_day,
        "hourly": ",".join(variable_names),
        "timezone": "UTC"
    }
    
    logger.debug("API parameters: %s", params)

    response = openmeteo.weather_api(url, params=params)[0]
    
    if not response or not response.Hourly():
        logger.warning("Invalid weather response received for coords=%s date=%s", coords, datatime_day)
        return {}
    
    hourly = response.Hourly()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),   #type: ignore  
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),     #type: ignore
        freq = pd.Timedelta(seconds = hourly.Interval()),           #type: ignore
        inclusive = "left"
    )}
    for i, var_name in enumerate(variable_names):
        variable = hourly.Variables(i)          #type: ignore
        if variable:
            hourly_data[var_name] = variable.ValuesAsNumpy()     #type: ignore
        else:
            logger.warning("Weather variable missing: index=%s name=%s", i, var_name)
            hourly_data[var_name] = None            #type: ignore
    
    
    hourly_dataframe = pd.DataFrame(data = hourly_data)
    weather_dict = hourly_dataframe[hourly_dataframe['date'] == datatime_utc].iloc[0].drop('date').to_dict()
    logger.debug("Weather info for instance: %s", weather_dict)
    
    return weather_dict
    



def get_data(zod_frames: ZodFrames) -> tuple[list[str], list[str]]:
    training_frames = zod_frames.get_split(constants.TRAIN)
    validation_frames = zod_frames.get_split(constants.VAL)

    training_frames = sorted(list(training_frames))
    validation_frames = sorted(list(validation_frames))
    logger.info("Number of training frames: %s", len(training_frames))
    logger.info("Number of validation frames: %s", len(validation_frames))
    return training_frames, validation_frames




def get_caracteristics(training_frames, zod_frames, num_frames: int, prev_frames_id: List[str]) -> tuple[dict, list]:
    logger.debug(
        "Starting get_caracteristics: total_training=%s num_frames=%s prev_ids=%s",
        len(training_frames), num_frames, len(prev_frames_id) if prev_frames_id is not None else 0,
    )
    total_frames_id = [zod_frames[frame_id].metadata.frame_id for frame_id in training_frames]
    frames_id = [fid for fid in total_frames_id if fid not in prev_frames_id]

    if len(frames_id) > num_frames:
        frames_id = frames_id[:num_frames]
    logger.info("Frames not in csv: %s", len(frames_id))
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
        "forward_acceleration": [],
        "lateral_acceleration": [],
        "forward_velocity": [],
        "lateral_velocity": [],
        "field_view_horizontal": [],
        "camera_distance_from_ground": [],
        "camera_pitch_angle": [],
        "distortion_magnitude": [],
        "camera_offset": [],
        #"focal_length": [],
    }   

    logger.debug("Fields initialized in data_dict: %s", list(data_dict.keys()))

    for id in tqdm(frames_id):
        logger.debug("Processing frame_id=%s", id)
        # Collect required metadata without appending yet 
        meta_country = zod_frames[id].metadata.country_code
        meta_time_of_day = zod_frames[id].metadata.time_of_day
        meta_lat = zod_frames[id].metadata.latitude
        meta_long = zod_frames[id].metadata.longitude
        meta_road_type = zod_frames[id].metadata.road_type
        meta_road_condition = zod_frames[id].metadata.road_condition
        meta_weather_label = zod_frames[id].metadata.scraped_weather
        meta_solar_angle = zod_frames[id].metadata.solar_angle_elevation
        meta_month = zod_frames[id].info.keyframe_time.month
        meta_hour = zod_frames[id].info.keyframe_time.hour  # In UTC+0, not local time
        meta_forward_acc = float(np.mean([a[0] * 3.6 for a in zod_frames[id].ego_motion.accelerations]))
        meta_lateral_acc = float(np.mean([a[1] * 3.6 for a in zod_frames[id].ego_motion.accelerations]))
        meta_forward_vel = float(np.mean([v[0] * 3.6 for v in zod_frames[id].ego_motion.velocities]))
        meta_lateral_vel = float(np.mean([v[1] * 3.6 for v in zod_frames[id].ego_motion.velocities]))
        cam = zod_frames[id].calibration.cameras[Camera.FRONT]

        try:
            weather_dict = get_weather_from_api(
                (meta_lat, meta_long),
                pd.to_datetime(str(zod_frames[id].info.keyframe_time), utc=True).round('h'),
            )
        except RetryError:
            logger.warning(
                "Weather API timed out after %ss for frame_id=%s; skipping frame",
                WEATHER_API_TIMEOUT_SECONDS,
                id,
            )
            continue
        except Exception as e:
            logger.exception("Weather API failed for frame_id=%s: %s", id, e)
            weather_dict = {}
        logger.debug("Weather dict keys for frame_id=%s: %s", id, list(weather_dict.keys()))

        # Append metadata only after weather call to allow skipping on timeout
        data_dict["country"].append(meta_country)
        data_dict["time_of_day"].append(meta_time_of_day)
        data_dict["lat"].append(meta_lat)
        data_dict["long"].append(meta_long)
        data_dict["road_type"].append(meta_road_type)
        data_dict["road_condition"].append(meta_road_condition)
        data_dict["weather"].append(meta_weather_label)
        data_dict["solar_angle_elevation"].append(meta_solar_angle)
        data_dict["month"].append(meta_month)
        data_dict["hour"].append(meta_hour)
        data_dict["forward_acceleration"].append(meta_forward_acc)
        data_dict["lateral_acceleration"].append(meta_lateral_acc)
        data_dict["forward_velocity"].append(meta_forward_vel)
        data_dict["lateral_velocity"].append(meta_lateral_vel)

        data_dict["field_view_horizontal"].append(float(cam.field_of_view[0]))
        T = cam.extrinsics.transform 
        data_dict["camera_distance_from_ground"].append(float(T[2, 3]))

        R = T[:3, :3]
        pitch = atan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        data_dict["camera_pitch_angle"].append(degrees(pitch))

        data_dict["distortion_magnitude"].append(float(np.linalg.norm(cam.distortion)))

        fx, fy = cam.intrinsics[0, 0], cam.intrinsics[1, 1]
        cx, cy = cam.intrinsics[0, 2], cam.intrinsics[1, 2]
        w, h = cam.image_dimensions[0], cam.image_dimensions[1]
        offset = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
        data_dict["camera_offset"].append(float(offset))

        #data_dict["focal_length"].append(float(fx / fy))

        
        for k, v in weather_dict.items():
            if k not in data_dict:
                data_dict[k] = []
            data_dict[k].append(v)
        
        zod_frame = zod_frames[id]
        image = zod_frame.get_image()
        logger.debug("Image fetched for frame_id=%s with shape=%s", id, getattr(image, "shape", None))
        
        use_fast = True
        #func = {"brisque":brisque_score, "quality":multimodal.CLIPImageQualityAssessment(), "brightness": multimodal.CLIPImageQualityAssessment(prompts=("brightness",)),"noisiness": multimodal.CLIPImageQualityAssessment(prompts=("noisiness",)), "sharpness": multimodal.CLIPImageQualityAssessment(prompts=("sharpness",)), "contrast": multimodal.CLIPImageQualityAssessment(prompts=("contrast",)), "complexity": multimodal.CLIPImageQualityAssessment(prompts=("complexity",))}
        func = {"laplacian":cv2.Laplacian, "clip":multimodal.CLIPImageQualityAssessment(prompts=("quality","brightness","noisiness", "sharpness", "contrast", "complexity"))}
        iqa = {fname: get_iqa(image, f) for fname, f in func.items()}
        logger.debug("IQA computed for frame_id=%s: keys=%s", id, list(iqa.keys()))
        for fname, f in iqa.items():
            if isinstance(f, dict):
                for prompt, piqa in f.items():
                    if prompt not in data_dict:
                        data_dict[prompt] = []
                    data_dict[prompt].append(piqa)
            else:
                if fname not in data_dict:
                    data_dict[fname] = []
                data_dict[fname].append(f)
        logger.debug("Data dict sizes snapshot for frame_id=%s: %s", id, {k: len(v) for k, v in data_dict.items() if isinstance(v, list)})
    return data_dict, frames_id

def to_csv(data_dict: dict[str, list], frames_id: list, resume: bool) -> None:
    base_dir = "/".join(DATA_DIR.split("/")[:-1])
    assert os.path.exists(base_dir), f"File {base_dir} does not exist, cant save csv"
    if resume:
        logger.info("Resuming write to %s", DATA_DIR)
        prev_df = pd.read_csv(DATA_DIR, index_col=0)
        current_df = pd.DataFrame(data=data_dict, index=frames_id)
        logger.debug("Previous DF shape=%s, Current DF shape=%s", prev_df.shape, current_df.shape)
        df = pd.concat([prev_df, current_df])
        df.to_csv(DATA_DIR)
        logger.info("Data saved to %s, total rows=%s", DATA_DIR, df.shape[0])
    else:
        logger.info("Writing new CSV to %s", DATA_DIR)
        df = pd.DataFrame(data=data_dict, index=frames_id)
        logger.debug("New DF shape=%s", df.shape)
        df.to_csv(DATA_DIR, index=True)
        logger.info("New CSV saved: %s", DATA_DIR)

def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("num_entries", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

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
