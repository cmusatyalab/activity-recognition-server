import numpy as np
import cv2
import queue
import logging
import imageio
from multiprocessing import Process, Queue
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2


NUM_INPUT_FRAMES = 32


logger = logging.getLogger(__name__)


class ActivityRecognitionEngine(cognitive_engine.Engine):
    ENGINE_NAME = 'activity_recognition'

    def handle(self, from_client):
        if from_client.payload_type != gabriel_pb2.PayloadType.VIDEO:
            return cognitive_engine.wrong_input_format_error(
                from_client.frame_id)

        reader = imageio.get_reader(from_client.payload, 'ffmpeg')
        W, H = reader.get_meta_data()['size']
        logger.info('Width: %s Height: %s', W, H)

        for frame_number in range(NUM_INPUT_FRAMES):
             cur_img = reader.get_data(frame_number)

             cv2.imencode("{}.jpg".format(frame_number), cur_img)
