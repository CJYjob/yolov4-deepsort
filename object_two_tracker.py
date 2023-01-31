import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detection_fast_reid as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video1', './data/video/section_cam_cut.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('video2', './data/video/number_recog_cut.mp4', 'path to input video2')
flags.DEFINE_string('output', './outputs/integrated.mp4', 'path to output1 video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.40, 'iou threshold')
flags.DEFINE_float('score', 0.40, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_string("video_mask1", './masking2.jpg', 'path to input video mask')
flags.DEFINE_string("video_mask2", None, 'path to input video mask')



def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/veriwild_dynamic.onnx'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric1 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    
    # initialize tracker
    tracker1 = Tracker(metric1)
    tracker2 = Tracker(metric2)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path1 = FLAGS.video1
    video_path2 = FLAGS.video2

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid1 = cv2.VideoCapture(int(video_path1))
    except:
        vid1 = cv2.VideoCapture(video_path1)

    vid2 = cv2.VideoCapture(video_path2)


    # get video mask
    if FLAGS.video_mask1:
        mask1 = cv2.imread(FLAGS.video_mask1)//255
    
    if FLAGS.video_mask2:
        mask2 = cv2.imread(FLAGS.video_mask2)//255
    
    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width_1 = int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_1 = int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid1.get(cv2.CAP_PROP_FPS))
        
        width_2 = int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_2 = int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width_1, height_1))

    frame_num = 0
    total_frame_num = min(vid1.get(cv2.CAP_PROP_FRAME_COUNT), vid2.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ratio = vid1.get(cv2.CAP_PROP_FRAME_COUNT) / vid2.get(cv2.CAP_PROP_FRAME_COUNT)
    # while video is running
    while True:
        return_value_1, frame1 = vid1.read()
        return_value_2, frame2 = vid2.read()        
        if (frame_num % 3 == 0) or (frame_num % 3 == 2):
            if return_value_2:
                return_value_2, frame2 = vid2.read()
        
        if return_value_1 and return_value_2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print(f'Frame #: {frame_num} / {total_frame_num}')
        
        if FLAGS.video_mask1:
            image_data1 = cv2.resize(frame1 * mask1, (input_size, input_size))
        else:
            image_data1 = cv2.resize(frame1, (input_size, input_size))
        
        if FLAGS.video_mask2:
            image_data2 = cv2.resize(frame2 * mask2, (input_size, input_size))
        else:
            image_data2 = cv2.resize(frame2, (input_size, input_size))
            
        image_data1 = image_data1 / 255.
        image_data1 = image_data1[np.newaxis, ...].astype(np.float32)
        
        image_data2 = image_data2 / 255.
        image_data2 = image_data2[np.newaxis, ...].astype(np.float32)
                
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data1)
            interpreter.invoke()
            pred1 = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            
            interpreter.set_tensor(input_details[0]['index'], image_data2)
            interpreter.invoke()
            pred2 = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                        
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes1, pred_conf1 = filter_boxes(pred1[1], pred1[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
                boxes2, pred_conf2 = filter_boxes(pred2[1], pred2[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes1, pred_conf1 = filter_boxes(pred1[0], pred1[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
                boxes2, pred_conf2 = filter_boxes(pred2[0], pred2[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))

        else:
            batch_data1 = tf.constant(image_data1)
            batch_data2 = tf.constant(image_data2)
            pred_bbox1 = infer(batch_data1)
            pred_bbox2 = infer(batch_data2)
            for key, value in pred_bbox1.items():
                boxes1 = value[:, :, 0:4]
                pred_conf1 = value[:, :, 4:]
            for key, value in pred_bbox2.items():
                boxes2 = value[:, :, 0:4]
                pred_conf2 = value[:, :, 4:]


        boxes1, scores1, classes1, valid_detections1 = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes1, (tf.shape(boxes1)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf1, (tf.shape(pred_conf1)[0], -1, tf.shape(pred_conf1)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        boxes2, scores2, classes2, valid_detections2 = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes2, (tf.shape(boxes2)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf2, (tf.shape(pred_conf2)[0], -1, tf.shape(pred_conf2)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects1 = valid_detections1.numpy()[0]
        bboxes1 = boxes1.numpy()[0]
        bboxes1 = bboxes1[0:int(num_objects1)]
        scores1 = scores1.numpy()[0]
        scores1 = scores1[0:int(num_objects1)]
        classes1 = classes1.numpy()[0]
        classes1 = classes1[0:int(num_objects1)]

        num_objects2 = valid_detections2.numpy()[0]
        bboxes2 = boxes2.numpy()[0]
        bboxes2 = bboxes2[0:int(num_objects2)]
        scores2 = scores2.numpy()[0]
        scores2 = scores2[0:int(num_objects2)]
        classes2 = classes2.numpy()[0]
        classes2 = classes2[0:int(num_objects2)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h1, original_w1, _ = frame1.shape
        bboxes1 = utils.format_boxes(bboxes1, original_h1, original_w1)

        original_h2, original_w2, _ = frame2.shape
        bboxes2 = utils.format_boxes(bboxes2, original_h2, original_w2)


        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox1 = [bboxes1, scores1, classes1, num_objects1]
        pred_bbox2 = [bboxes2, scores2, classes2, num_objects2]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names1 = []
        deleted_indx1 = []
        for i in range(num_objects1):
            class_indx = int(classes1[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx1.append(i)
            else:
                names1.append(class_name)
        names1 = np.array(names1)
        count1 = len(names1)
        if FLAGS.count:
            cv2.putText(frame1, "Objects being tracked: {}".format(count1), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count1))
        # delete detections that are not in allowed_classes
        bboxes1 = np.delete(bboxes1, deleted_indx1, axis=0)
        scores1 = np.delete(scores1, deleted_indx1, axis=0)

        names2 = []
        deleted_indx2 = []
        for i in range(num_objects2):
            class_indx = int(classes2[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx2.append(i)
            else:
                names2.append(class_name)
        names2 = np.array(names2)
        count2 = len(names2)
        if FLAGS.count:
            cv2.putText(frame2, "Objects being tracked: {}".format(count2), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count2))
        # delete detections that are not in allowed_classes
        bboxes2 = np.delete(bboxes2, deleted_indx2, axis=0)
        scores2 = np.delete(scores2, deleted_indx2, axis=0)

        # encode yolo detections and feed to tracker
        features1 = encoder(frame1, bboxes1)
        detections1 = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes1, scores1, names1, features1)]
        features2 = encoder(frame2, bboxes2)
        detections2 = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes2, scores2, names2, features2)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs1 = np.array([d.tlwh for d in detections1])
        scores1 = np.array([d.confidence for d in detections1])
        classes1 = np.array([d.class_name for d in detections1])
        indices1 = preprocessing.non_max_suppression(boxs1, classes1, nms_max_overlap, scores1)
        detections1 = [detections1[i] for i in indices1]       

        boxs2 = np.array([d.tlwh for d in detections2])
        scores2 = np.array([d.confidence for d in detections2])
        classes2 = np.array([d.class_name for d in detections2])
        indices2 = preprocessing.non_max_suppression(boxs2, classes2, nms_max_overlap, scores2)
        detections2 = [detections2[i] for i in indices2]       

        # Call the tracker
        tracker1.predict()
        tracker1.update(detections1)
        tracker2.predict()
        tracker2.update(detections2)

        # update tracks
        for track in tracker1.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame1, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame1, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        frame2_resized = cv2.resize(frame2, (int(width_2//4), int(height_2//4)))

        # update tracks
        for track in tracker2.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
                       
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame2_resized, (int(bbox[0]//4), int(bbox[1]//4)), (int(bbox[2]//4), int(bbox[3]//4)), color, 2)
            cv2.rectangle(frame2_resized, (int(bbox[0]//4), int(bbox[1]//4-30)), (int(bbox[0]//4)+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1]//4)), color, -1)
            cv2.putText(frame2_resized, class_name + "-" + str(track.track_id),(int(bbox[0]//4), int(bbox[1]//4-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))


        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        
        
        frame1_and_2 = frame1.copy()
        
        frame1_and_2[:int(height_2//4), int(width_1-width_2//4):, :] = frame2_resized
        
        
        result = np.asarray(frame1_and_2)
        # result2 = np.asarray(frame2)

        result = cv2.cvtColor(frame1_and_2, cv2.COLOR_RGB2BGR)
        # result2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
