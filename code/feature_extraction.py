import numpy as np
import cv2 as cv
import tensorflow as tf
from skimage import feature

from data_extraction import *


def get_video_data(event_data, bbox_data, video_id):
    return event_data[event_data["#YoutubeId"] == video_id], bbox_data[bbox_data["Youtube ID"] == video_id]


model_fn = 'tensorflow_inception_graph.pb'

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = t_input-imagenet_mean
tf.import_graph_def(graph_def, {'input':t_preprocessed})

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)


def preprocess_frame(frame, new_shape, kernel_size, sigma):
    """Preprocess frame befor feeding the network
    
    Parameters
    ==========
    frame: np.float32
        image
    new_shape: tuple
        target resized image
    kernel_size: tuple
        kernel size for gaussian blur
    sigma: float
        sigma for gaussian blur
    """
    
    if 0 in frame.shape:
        return np.zeros(new_shape+ (3,))
    
    blured = cv.GaussianBlur(frame, kernel_size, sigma)
    resized = cv.resize(blured, new_shape)
    
    return resized


def extract_frame_features(frame, t_obj, new_shape, kernel_size, sigma):
    """Extract frame features from network object
    
    Paramters
    =========
    frame: np.float32
        image to be extracted features assumed in shape [image_height, image_width, 3]
    t_obj: tf.Tensor
        tensorflow object to be evaluated
    new_shape: tuple
        target resized image
    kernel_size: tuple
        kernel size for gaussian blur
    sigma: float
        sigma for gaussian blur
    """
    preprocessed = preprocess_frame(frame, new_shape, kernel_size, sigma)
    
    preprocessed = np.expand_dims(preprocessed, 0)
    
    features = t_obj.eval({t_input: preprocessed})
    
    return features.squeeze()

def extract_frame_features_batch(frames, t_obj, new_shape, kernel_size, sigma):
    """Extract frame features from network object
    
    Paramters
    =========
    frame: np.float32
        image to be extracted features assumed in shape [image_height, image_width, 3]
    t_obj: tf.Tensor
        tensorflow object to be evaluated
    new_shape: tuple
        target resized image
    kernel_size: tuple
        kernel size for gaussian blur
    sigma: float
        sigma for gaussian blur
    """
    preprocessed = np.array([preprocess_frame(frame, new_shape, kernel_size, sigma) for frame in frames])
    
    
    features = t_obj.eval({t_input: preprocessed})
    
    return features.squeeze()


def extract_appeareance_player(player_region, t_obj, new_shape, kernel_size, sigma, pool_size, strides):
    """Extract appearance feature
    
    Parameters
    ==========
    player_region: np.float32
        image of player region
    t_obj: tf.Tensor
        tensorflow object to be evaluated
    new_shape: tuple
        target resized image
    kernel_size: tuple
        kernel size for gaussian blur
    sigma: float
        sigma for gaussian blur
    pool_size: tuple
        shape of the pool size for average pooling
    strides: tuple
        stride
    """
    
    # TODO: Implement region proposals
    preprocessed = preprocess_frame(player_region, new_shape, kernel_size, sigma)
    preprocessed = np.expand_dims(preprocessed, 0)
    
    t_avg_pool = tf.layers.average_pooling2d(t_obj, pool_size=pool_size, strides=strides, padding="VALID", name="avg_pool")
    features = t_avg_pool.eval({t_input: preprocessed})
    
    return features.reshape(-1)

def extract_appeareance_player_batch(player_regions, t_avg_pool, new_shape, kernel_size, sigma):
    """Extract appearance feature
    
    Parameters
    ==========
    player_region: np.float32
        image of player region
    t_obj: tf.Tensor
        tensorflow object to be evaluated
    new_shape: tuple
        target resized image
    kernel_size: tuple
        kernel size for gaussian blur
    sigma: float
        sigma for gaussian blur
    pool_size: tuple
        shape of the pool size for average pooling
    strides: tuple
        stride
    """
    
    # TODO: Implement region proposals
    preprocessed = np.array([preprocess_frame(player_region, new_shape, kernel_size, sigma) for player_region in player_regions])
    
    features = t_avg_pool.eval({t_input: preprocessed})
    
    return features.reshape(len(player_regions), -1)

def pyramid(img, scale, kernel_size, sigma, lower_bound=0):
    yield img
    height, width = img.shape[:2]
    height, width = height // scale, width // scale

    while height > lower_bound and width > lower_bound:
        img = cv.GaussianBlur(img, kernel_size, sigma, sigma)
        img = cv.resize(img, (width, height))
        yield img
        height, width = height // scale, width // scale


def extract_spatial_feature_from_player_region(player_region, shape, scale, kernel_size, sigma, lower_bound, orientations, pixels_per_cell, cells_per_block, block_norm):
    if 0 in player_region.shape:
        player_region = np.zeros(shape)
        
    features = [feature.hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm=block_norm) for img in pyramid(cv.resize(player_region, shape), scale, kernel_size, sigma, lower_bound)]
    return np.hstack(features)

def extract_spatial_feature_from_player_region_batch(player_regions,
                                                      network_shape=(224, 224), 
                                                      kernel_size=(5, 5), 
                                                      sigma=0.1, 
                                                      local_shape=(32, 32), 
                                                      scale=2, 
                                                      lower_bound=15, 
                                                      orientations=18,
                                                      pixels_per_cell=(4, 4),
                                                      cells_per_block=(1, 1),
                                                      block_norm="L1"):
    """Extract appearance feature
    
    Parameters
    ==========
    player_region: np.float32
        image of player region
    t_obj: tf.Tensor
        tensorflow object to be evaluated
    new_shape: tuple
        target resized image
    kernel_size: tuple
        kernel size for gaussian blur
    sigma: float
        sigma for gaussian blur
    pool_size: tuple
        shape of the pool size for average pooling
    strides: tuple
        stride
    """
    
    # TODO: Implement region proposals
    features = np.array([extract_spatial_feature_from_player_region(player_region,
                                                                 local_shape,
                                                                 scale,
                                                                 kernel_size,
                                                                 sigma,
                                                                 lower_bound,
                                                                 orientations,
                                                                 pixels_per_cell,
                                                                 cells_per_block,
                                                                 block_norm) for player_region in player_regions])
    
    
    return features



def extract_localized_features_for_person(player_region, 
                                          layer="mixed4a", 
                                          network_shape=(224, 224), 
                                          kernel_size=(5, 5), 
                                          sigma=0.1, 
                                          pool_size=(7, 7), 
                                          strides=(7, 7), 
                                          local_shape=(32, 32), 
                                          scale=2, 
                                          lower_bound=15, 
                                          orientations=18,
                                          pixels_per_cell=(4, 4),
                                          cells_per_block=(1, 1),
                                          block_norm="L1"):
    """Extract localized feature for each player
    
    Parameters
    ==========
    player_region: np.float32
        img to be extracted features
    layer: str
        layer to be used from network
    network_shape: tuple
        image shape valid for network
    kernel_size: tuple
        gaussian kernel size
    sigma: float
        sigma to be used for gaussian blur
    pool_size: tuple
        pool size for spatial features
    strides: tuple
        stride for avg pooling
    local_shape: tuple
        shape for local spatial histograms
    scale: int
        scale factor for pyramid
    lower_bound: float
        to be used in pyramid
    orientations: int
        number of orientations used to calculate hog features
    pixels_per_cell: tuple
        pixels per cell
    cells_per_block: tuple
        cells per block
    block_norm: str
        block normalization method
    """
    
    t_obj = T(layer)

    appearence_feature = extract_appeareance_player(player_region,
                                                    t_obj,
                                                    new_shape=network_shape,
                                                    kernel_size=kernel_size,
                                                    sigma=sigma,
                                                    pool_size=pool_size,
                                                    strides=strides)
    
    spatial_feature = extract_spatial_feature_from_player_region(player_region,
                                                                 local_shape,
                                                                 scale,
                                                                 kernel_size,
                                                                 sigma,
                                                                 lower_bound,
                                                                 orientations,
                                                                 pixels_per_cell,
                                                                 cells_per_block,
                                                                 block_norm)
    
    return np.hstack([appearence_feature, spatial_feature])


def extract_features(event_data, bbox_data, video_id, events_paths_list, width, height):
    video_events, video_bbox = get_video_data(event_data, bbox_data, video_id)
    # Events paths list must be given in order by end times of events
    event_map = create_events_map(event_data=video_events, events_paths_list=events_paths_list)
    reverse_event_map = {value: key for key, value in event_map.items()}

    event_bboxtimes_map = create_eventbboxtimes_map(video_bbox, event_map)

    layer = "avgpool0"

    events = []
    for event_path in events_paths_list:
        event = []
        for frame, player_dict in extract_players_from_frames(event_path, event_bboxtimes_map, reverse_event_map[event_path], video_bbox, width, height):
            t_obj = T(layer)
            ft = extract_frame_features(frame=frame, kernel_size=(5, 5), new_shape=(224, 224), sigma=0.1, t_obj=t_obj)
            pt = []
            for player in player_dict:
                with tf.Session(graph=graph) as sess:
                    pt.append(extract_localized_features_for_person(player_dict[player]))

            event.append((ft, pt))
        events.append(event)

    return events

def extract_features_batch(event_data, bbox_data, video_id, events_paths_list, width, height):
    
    video_events, video_bbox = get_video_data(event_data, bbox_data, video_id)
    event_map = create_events_map(event_data=video_events, events_paths_list=events_paths_list)
    reverse_event_map = {value: key for key, value in event_map.items()}

    event_bboxtimes_map = create_eventbboxtimes_map(video_bbox, event_map)

    layer = "avgpool0"
    t_obj_frame = T(layer)

    layer = "mixed4a"
    t_obj_player = T(layer)
    strides = (7, 7)
    pool_size = (7, 7) 
    t_avg_pool = tf.layers.average_pooling2d(t_obj_player, pool_size=pool_size, strides=strides, padding="VALID", name="avg_pool")    

    features = []
    for event_path in events_paths_list:
        frames, player_dicts = zip(*extract_players_from_frames(event_path, event_bboxtimes_map, reverse_event_map[event_path], video_bbox, width, height))
        ft = extract_frame_features_batch(frames=frames, kernel_size=(5, 5), new_shape=(224, 224), sigma=0.1, t_obj=t_obj_frame)
        pts = []

        for player_dict in player_dicts:
            player_ids, player_regions = zip(*player_dict.items())
            pt_appereance = extract_appeareance_player_batch(player_regions, t_avg_pool, kernel_size=(5, 5), new_shape=(224, 224), sigma=0.1)
            pt_spatial = extract_spatial_feature_from_player_region_batch(player_regions)
            pt = np.concatenate((pt_appereance, pt_spatial), axis=1)
            pt = {player_id: pt_i for player_id, pt_i in zip(player_ids, pt)}

            pts.append(pt)


        features.append(list(zip(list(ft), pts)))

    video_events = video_events.sort_values(by="EventEndTime")
    labels = video_events.EventLabel.tolist()


    return list(zip(features, labels))


