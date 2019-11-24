import numpy as np
import cv2
import imageio
import logging
import os
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2

import object_detection.object_detector as obj
import action_detection.action_detector as act


NUM_INPUT_FRAMES = 32
WIDTH = 640
HEIGHT = 480
OBJ_DETECTION_MODEL = 'ssd_mobilenet_v2_coco_2018_03_29'
CKPT_NAME = 'model_ckpt_soft_attn_pooled_cosine_drop_ava-130'
MAIN_FOLDER = './'
MAX_ACTORS = 14
PRINT_TOP_K = 5
ACTION_TH = 0.2
COMPRESSION_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 67]

PROJECTOR_PREAMBLE = 'Playing extra sound for projector.'

np.random.seed(10)
# get darker colors for bboxes and use white text
COLORS = np.random.randint(0, 100, [1000, 3])


logger = logging.getLogger(__name__)


def gen_text_result(text):
    result_wrapper = gabriel_pb2.ResultWrapper()
    
    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.TEXT
    result.engine_name = ActivityRecognitionEngine.ENGINE_NAME
    result.payload = text.encode(encoding="utf-8")
    
    result_wrapper.results.append(result)
    return result_wrapper


class ActivityRecognitionEngine(cognitive_engine.Engine):
    ENGINE_NAME = 'activity_recognition'

    def __init__(self):
        obj_detection_graph = os.path.join(
            "object_detection", "weights", OBJ_DETECTION_MODEL,
            "frozen_inference_graph.pb")
        self.obj_detector = obj.Object_Detector(obj_detection_graph)

        self.act_detector = act.Action_Detector(
            'soft_attn', timesteps=NUM_INPUT_FRAMES)
        crop_in_tubes = self.act_detector.crop_tubes_in_tf(
            [NUM_INPUT_FRAMES, HEIGHT, WIDTH, 3])
        (self.input_frames, self.temporal_rois, self.temporal_roi_batch_indices,
         self.cropped_frames) = crop_in_tubes
        self.rois, self.roi_batch_indices, self.pred_probs = (
            self.act_detector.define_inference_with_placeholders_noinput(
                self.cropped_frames))

        ckpt_path = os.path.join(
            MAIN_FOLDER, 'action_detection', 'weights', CKPT_NAME)
        self.act_detector.restore_model(ckpt_path)

    def handle(self, from_client):
        if from_client.payload_type != gabriel_pb2.PayloadType.VIDEO:
            return cognitive_engine.wrong_input_format_error(
                from_client.frame_id)

        reader = imageio.get_reader(from_client.payload, 'ffmpeg')

        assert reader.get_length() > NUM_INPUT_FRAMES, (
            'Video only had {} frames'.format(reader.get_length()))
        
        w, h = reader.get_meta_data()['size']
        assert w == WIDTH, 'Input width was {}'.format(w)
        assert h == HEIGHT, 'Input height was {}'.format(h)

        img_batch = [
            reader.get_data(frame_number)
            for frame_number in range(NUM_INPUT_FRAMES)
        ]
        expanded_img = np.stack(img_batch, axis=0)

        detection_list = self.run_obj_detector(expanded_img)

        self.tracker = obj.Tracker(timesteps=NUM_INPUT_FRAMES)
        self.update_tracker(img_batch, detection_list)
        num_actors = len(self.tracker.active_actors)

        if num_actors == 0:
            logger.info('No active actors')
            result_wrapper = gen_text_result('{} No people detected'.format(PROJECTOR_PREAMBLE))
        else:
            act_results = self.run_act_detector(num_actors)
            prob_dict = self.build_prob_dict(act_results, num_actors)
            result_wrapper = self.gen_result_wrapper(img_batch[-1], prob_dict)

        result_wrapper.frame_id = from_client.frame_id
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS

        return result_wrapper

    def run_obj_detector(self, expanded_img):
        return self.obj_detector.detect_objects_in_np(expanded_img)

    def update_tracker(self, img_batch, detection_list):
        for frame_number in range(NUM_INPUT_FRAMES):
            cur_img = img_batch[frame_number]
            detection_info = [info[frame_number] for info in detection_list]
            self.tracker.update_tracker(detection_info, cur_img)

    def run_act_detector(self, num_actors):
        cur_input_sequence = np.expand_dims(np.stack(self.tracker.frame_history[-NUM_INPUT_FRAMES:], axis=0), axis=0)

        rois_np, temporal_rois_np = self.tracker.generate_all_rois()
        if num_actors > MAX_ACTORS:
            num_actors = MAX_ACTORS
            rois_np = rois_np[:MAX_ACTORS]
            temporal_rois_np = temporal_rois_np[:MAX_ACTORS]

        feed_dict = {
            self.input_frames : cur_input_sequence,
            self.temporal_rois : temporal_rois_np,
            self.temporal_roi_batch_indices : np.zeros(num_actors),
            self.rois : rois_np,
            self.roi_batch_indices : np.arange(num_actors)
        }
        run_dict = { 'pred_probs' : self.pred_probs }

        return self.act_detector.session.run(run_dict, feed_dict=feed_dict)

    def build_prob_dict(self, act_results, num_actors):
        prob_dict = {}
        for bounding_box in range(num_actors):
            act_probs = act_results['pred_probs'][bounding_box]
            order = np.argsort(act_probs)[::-1]
            cur_actor_id = self.tracker.active_actors[bounding_box]['actor_id']
            logger.info('Person %d', cur_actor_id)
            cur_results = []
            for pp in range(PRINT_TOP_K):
                logger.info('\t %s: %.3f', act.ACTION_STRINGS[order[pp]], act_probs[order[pp]])
                cur_results.append((act.ACTION_STRINGS[order[pp]], act_probs[order[pp]]))
            prob_dict[cur_actor_id] = cur_results

        return prob_dict

    def gen_result_wrapper(self, disp_img, prob_dict):
        H, W, C = disp_img.shape

        descriptions = []
        for ii in range(len(self.tracker.active_actors)):
            cur_actor = self.tracker.active_actors[ii]
            actor_id = cur_actor['actor_id']
            cur_act_results = prob_dict[actor_id] if actor_id in prob_dict else []
            cur_box = cur_actor['all_boxes'][-1]
            cur_class = 1

            top, left, bottom, right = cur_box

            left = int(W * left)
            right = int(W * right)
            
            top = int(H * top)
            bottom = int(H * bottom)

            label = obj.OBJECT_STRINGS[cur_class]['name']
            person_identifier = '{} {}'.format(label.capitalize(), actor_id)

            results_to_report = [
                cur_act_result[0] for cur_act_result in cur_act_results
                if cur_act_result[1] > ACTION_TH
            ]

            if len(results_to_report) > 0:
                description = '{} was labeled with the following {} lables: {}.'.format(
                    person_identifier, len(results_to_report), ', '.join(results_to_report))
            else:
                description = 'Found no actions for {}.'.format(person_identifier)
            descriptions.append(description)

            raw_colors = COLORS[actor_id]
            rect_color = tuple(int(raw_color) for raw_color in raw_colors)
            text_color = tuple(255-color_value for color_value in rect_color)

            cv2.rectangle(disp_img, (left,top), (right,bottom), rect_color, 3)

            font_size =  max(0.5,(right - left)/50.0/float(len(person_identifier)))
            cv2.rectangle(disp_img, (left, top-int(font_size*40)), (right,top), rect_color, -1)
            cv2.putText(disp_img, person_identifier, (left, top-12), 0, font_size, text_color, 1)

        description = ' '.join(descriptions)
        result_wrapper = gen_text_result(PROJECTOR_PREAMBLE + description)
        
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.engine_name = ActivityRecognitionEngine.ENGINE_NAME
        _, jpeg_img = cv2.imencode(".jpg", disp_img, COMPRESSION_PARAMS)
        result.payload = jpeg_img.tostring()
        result_wrapper.results.append(result)
        
        return result_wrapper
