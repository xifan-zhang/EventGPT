import torch
from dataset.constants import EVENT_TOKEN_INDEX, DEFAULT_EVENT_TOKEN, DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN, EVENT_PLACEHOLDER, DEFAULT_EVENT_PATCH_TOKEN
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import sys

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_event_images_list(event_npy, n):
    x, y, p, t = event_npy['x'], event_npy['y'], event_npy['p'], event_npy['t']

    total_events = len(t)
    events_per_image = total_events // n

    event_image_list = []

    for i in range(n):
        start_idx = i * events_per_image
        end_idx = (i + 1) * events_per_image if i < n - 1 else total_events

        x_part = x[start_idx:end_idx]
        y_part = y[start_idx:end_idx]
        p_part = p[start_idx:end_idx]

        event_img = generate_event_image(x_part, y_part, p_part)

        event_image_list.append(event_img)

    return event_image_list

def check_EventStream_length(start_time, end_time):
    if end_time - start_time >= 100000:
        raise Exception("Apologies, EventGPT currently does not support Event Streams exceeding 100ms. Please stay tuned for updates in our future versions.")

def tokenizer_event_token(prompt, tokenizer, event_token_index=EVENT_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<event>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [event_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def generate_event_image(x, y, p):
    height, width = y.max() + 1, x.max() + 1 
    event_image = np.ones((height, width, 3), dtype=np.uint8) * 255 # one image?

    for x_, y_, p_ in zip(x, y, p):
        if p_ == 0:
            event_image[y_, x_] = np.array([0, 0, 255])  # Blue for negative polarity
        else:
            event_image[y_, x_] = np.array([255, 0, 0])  # Red for positive polarity

    return event_image

def split_event_by_time(event_npy, time_interval=50000):
        """
        Split event data into time intervals (default 50ms).
        
        :param event_npy: Dictionary containing event data with keys 'p', 't', 'x', 'y'.
        :param time_interval: Time interval for splitting in microseconds, default is 50ms (50,000 microseconds).
        :return: A list of dictionaries where each dictionary corresponds to a time interval with the split data.
        """
        # Extract data from the event_npy dictionary
        p = event_npy['p']
        t = event_npy['t']
        x = event_npy['x']
        y = event_npy['y']

        # Calculate the time bin for each timestamp based on the given time interval
        time_bins = (t // time_interval) * time_interval

        # Get the unique time bins (intervals)
        unique_bins = np.unique(time_bins)

        # Split the data according to the time bins
        split_data = [
            {
                'p': p[time_bins == bin],
                't': t[time_bins == bin],
                'x': x[time_bins == bin],
                'y': y[time_bins == bin]
            }
            for bin in unique_bins
        ]

        return split_data


def process_event_data(event_frame_path, event_processor, device):
    event_npy = np.load(event_frame_path, allow_pickle=True)
    event_npy = np.array(event_npy).item()

    if event_npy['t'].max() - event_npy['t'].min() >= 100000:
        raise Exception("Apologies, EventGPT currently does not support Event Streams exceeding 100ms. "
                        "Please stay tuned for updates in our future versions.")

    event_img_list = get_event_images_list(event_npy, 5)
    event_image_size = list(event_img_list[0].shape[:2])

    event_list = []
    for event in event_img_list:
        event = event_processor(event, return_tensors='pt')['pixel_values'][0]
        event = event.to(device, dtype=torch.bfloat16)
        event_list.append(event)

    return event_image_size, event_list
