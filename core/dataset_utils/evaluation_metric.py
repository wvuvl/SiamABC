#Command line tool for evaluating tracking
'''

source: https://github.com/versey-sherry/track_eval

Input
--prediction_dir path to the txt files of predicted outputs
--video_dir path to the training sequence with gt
--results_dir folder to save the results
--writeout True if video output is needed

Single object tracking evaluation metrics
Accuracy: The accuracy is the average overlap between the predicted [Done]
and ground truth bounding boxes during successful tracking periods. 
Robustness: The robustness measures how many times the tracker loses the target (fails) during tracking.
Expected Average Overlap (EAO) is measured by averaging the accuracy over all videos

All video dirs should be in the same dir, to process LaSOT
find . -maxdepth 2  -print -exec mv {} . \;
'''

'''
example
python evaluation.py --prediction_dir result_example/sot_results/TB-100/goturn \
--video_dir TB-100 \
--gt_name groundtruth_rect.txt \
--threshold 0.5 \
--writeout True

'''

import os
import cv2
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np




def load_annotation_file_otb(ann_file):
    '''
    @alov_sub_dir: subdirectory of images directory
    @ann_file: annotation file to fetch the bounding boxes from
    @ext: image extension
    '''

    bboxes = []
    frame_num = 0
    with open(ann_file) as f:
        data = f.read().rstrip().split('\n')
        for bb in data:
            if ',' in bb: x1, y1, w, h = [int(i) for i in bb.split(',')]
            else: x1, y1, w, h = [int(float(i)) for i in bb.split()]
            bboxes.append([x1, y1, x1+w, y1+h, frame_num])
            frame_num+=1

    return bboxes

def load_annotation_file(ann_file):
    '''
    @alov_sub_dir: subdirectory of images directory
    @ann_file: annotation file to fetch the bounding boxes from
    @ext: image extension
    '''

    bboxes = []
    with open(ann_file) as f:
        data = f.read().rstrip().split('\n')
        for bb in data:
            frame_num, ax, ay, bx, by, cx, cy, dx, dy = [float(i) for i in bb.split()]
            frame_num = int(frame_num)

            x1 = int(min(ax, min(bx, min(cx, dx))) - 1)
            y1 = int(min(ay, min(by, min(cy, dy))) - 1)
            x2 = int(max(ax, max(bx, max(cx, dx))) - 1)
            y2 = int(max(ay, max(by, max(cy, dy))) - 1)

            bboxes.append([x1, y1, x2, y2, frame_num])

    return bboxes
        

# compute IoU between prediction and ground truth, bounding box input is x1,y1,x2,y2
def compute_iou(prediction, gt):
    #ensure the bounding boxes exist
    assert(prediction[0] <= prediction[2])
    assert(prediction[1] <= prediction[3])
    assert(gt[0] <= gt[2])
    assert(gt[1] <= gt[3])

    #intersection rectangule
    xA = max(prediction[0], gt[0])
    yA = max(prediction[1], gt[1])
    xB = min(prediction[2], gt[2])
    yB = min(prediction[3], gt[3])

    #compute area of intersection
    interArea = max(0, xB-xA + 1) * max(0, yB - yA + 1)

    #compute the area of the prection and gt
    predictionArea = (prediction[2] - prediction[0] +1) * (prediction[3] - prediction[1] +1)
    gtArea = (gt[2] - gt[0] + 1) * (gt[3]-gt[1]+1)

    #computer intersection over union
    iou = interArea / float(predictionArea+gtArea-interArea)
    return iou

#mask iou is computed via sklearn
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score

#Process single object ground truth and predition
#ground truth format <x>, <y>, <w>, <h>
#predition format <x>, <y>, <w>, <h>
#return a ditionary with frame as key and a list of [<x>, <y>, <w>, <h>] presented in the frame
def process_single(txt_file):
    info = defaultdict(list)
    a = 1
    with open(txt_file) as file:
        for line in file:
            try:
                info[a].append(list(eval(line)))
            except:
                line = [int(float(item)) for item in line.split()]
                info[a].append(line)
            a+=1
    return info



# Evaluation for one video
# output video name, accuracy and robustness
'''
video_dir = 'gt_example/TB-100'
prediction_dir = 'result_example/sot_results/TB-100/goturn'
save_dir = './results'
prediction = os.path.join(prediction_dir, 'Biker.txt')
gt = os.path.join(video_dir, 'Biker', 'groundtruth_rect.txt')
'''
def single_eval(prediction, gt, save_dir, threshold, img_dir, vid_name, writeout=False):
    
    prediction_dict =  load_annotation_file_otb(prediction)
    gt_dict = load_annotation_file_otb(gt)


    # prediction_dict = prediction # load_annotation_file_otb(prediction)
    # gt_dict = gt #load_annotation_file_otb(gt)

    if os.path.exists(save_dir):
        print('Save directory already exists')
    else:
        os.makedirs(save_dir)
        print('Making new save dir')

    #preparation for writing out the video
    if writeout:
        images = sorted([file for file in os.listdir(img_dir) if '.jpg' in file or '.jpeg' in file])
        img = cv2.imread(os.path.join(img_dir, images[0]))
        height, width, _ = img.shape
        size = (width, height)
        file_name = os.path.join(save_dir, vid_name+'.mp4')
        print('Video saved to', file_name)
        out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

    acc_list = []
    robustness = 0
    frame_num = 0

    #loop through every frame in ground truth
    while frame_num < min(len(gt_dict), len(prediction_dict)):
        pred_info = prediction_dict[frame_num]
        gt_info = gt_dict[frame_num]

        prediction = [pred_info[0], pred_info[1], pred_info[2], pred_info[3]]
        gt = [gt_info[0], gt_info[1], gt_info[2], gt_info[3]]

        acc = compute_iou(prediction, gt)
        acc_list.append(acc)

        if acc > threshold:
            robustness +=1
        #print(prediction)
        #print(gt)
        
        #output videos
        if writeout:
            img = cv2.imread(os.path.join(img_dir, images[gt_info[4]]))
            #print(images[frame_num-1])
            #ploting prediction with red box
            img = cv2.rectangle(img, (prediction[0], prediction[1]), (prediction[2], prediction[3]), (0, 0, 255), 2)
            #ploting ground truth with green box
            img = cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0,255,0), 2)
            #Put stats on frame
            text = 'Frame {}: IoU is {}%'.format(frame_num+1,round((acc *100),2))
            img = cv2.putText(img, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA) 
            out.write(img)
        
        frame_num+=1
    
    if writeout:
        out.release()

    if len(acc_list) >0:
        accuracy = sum(acc_list)/len(acc_list)
    else:
        print('No IoU')

    if robustness >0:
        robustness = robustness/len(gt_dict)
        
    return accuracy, robustness


def main():
    parser = argparse.ArgumentParser(description='tracker evaluation')
    parser.add_argument('--prediction_dir', type=str, help = 'path to the txt files of the predicted outputs')
    parser.add_argument('--video_dir', type=str, help = 'path to the directory that contains image sequence and gt')
    parser.add_argument('--gt_name', type=str, help= "naming rules for ground truth")
    parser.add_argument('--save_dir', type=str, default= './results')
    parser.add_argument('--threshold', type=float, default = 0.5, help= "threshold for correctly tracked")
    parser.add_argument('--writeout', type=bool, default = False, help= "True if there is video output needed")
    args= parser.parse_args()

    #list all the videos to be evaluated
    video_list = [folder for folder in os.listdir(args.video_dir) if '.' not in folder]

    print('The video(s) to be evaluated', video_list)
    # A list to hold the summary for reporting
    output_list = []

    accuracy_list = []
    robustness_list = []
    

    for folder in video_list:
        prediction = os.path.join(args.prediction_dir, folder+'.txt')
        print('Prediction file is', prediction)
        gt = os.path.join(args.video_dir, folder, args.gt_name)
        print('Ground truth is', gt)

        
        name, accuracy, robustness = single_eval(prediction, gt, args.save_dir, args.threshold, writeout = args.writeout)
        
        accuracy_list.append(accuracy)
        robustness_list.append(robustness)
        text = '{}\nAccuracy is {}%, robustness is {}%'.format(name, round(accuracy*100, 2), round (robustness*100, 2))
        print(text)
        output_list.append(text)
        

    if len(accuracy_list) >0:
        total_accuracy = sum(accuracy_list)/len(accuracy_list)
    else:
        print('No accuracy')
    
    if len(robustness_list) >0:
        total_robustness = sum(robustness_list)/len(robustness_list)
    else:
        print('No robustness')

    text = '{}: Accuracy is {}% Robustness is {}%'.format(
        args.video_dir.split('/')[-1], 
        round(total_accuracy*100, 2), 
        round(total_robustness *100, 2))
    #print(text)
    output_list.append(text)


    #writing out the results
    with open(os.path.join(args.save_dir, args.prediction_dir.split('/')[-1]+'_results.txt'), 'w') as file:
        for item in output_list:
            file.write(''.join([item, '\n']))
            
def plot_curves(nbins_iou, nbins_ce, succ_curve, prec_curve, succ_score, prec_score, save_path):
        

    succ_file = os.path.join(save_path, 'success_plots.png')
    prec_file = os.path.join(save_path, 'precision_plots.png')
    
    # markers
    markers = ['-', '--', '-.']
    markers = [c + m for m in markers for c in [''] * 10]

    # plot success curves
    thr_iou = np.linspace(0, 1, nbins_iou)
    fig, ax = plt.subplots()
    lines = []
    legends = []
    for i, name in enumerate(['FEAR']):
        line, = ax.plot(thr_iou,
                        succ_curve,
                        markers[i % len(markers)])
        lines.append(line)
        legends.append('%s: [%.3f]' % (name, succ_score))
    matplotlib.rcParams.update({'font.size': 7.4})
    legend = ax.legend(lines, legends, loc='center left',
                        bbox_to_anchor=(1, 0.5))

    matplotlib.rcParams.update({'font.size': 9})
    ax.set(xlabel='Overlap threshold',
            ylabel='Success rate',
            xlim=(0, 1), ylim=(0, 1),
            title='Success plots of OPE')
    ax.grid(True)
    fig.tight_layout()
    
    fig.savefig(succ_file,
                bbox_extra_artists=(legend,),
                bbox_inches='tight',
                dpi=300)

    # plot precision curves
    thr_ce = np.arange(0, nbins_ce)
    fig, ax = plt.subplots()
    lines = []
    legends = []
    for i, name in enumerate(['FEAR']):
        line, = ax.plot(thr_ce,
                        prec_curve,
                        markers[i % len(markers)])
        lines.append(line)
        legends.append('%s: [%.3f]' % (name, prec_score))
    matplotlib.rcParams.update({'font.size': 7.4})
    legend = ax.legend(lines, legends, loc='center left',
                        bbox_to_anchor=(1, 0.5))
    
    matplotlib.rcParams.update({'font.size': 9})
    ax.set(xlabel='Location error threshold',
            ylabel='Precision',
            xlim=(0, thr_ce.max()), ylim=(0, 1),
            title='Precision plots of OPE')
    ax.grid(True)
    fig.tight_layout()
    
    fig.savefig(prec_file, dpi=300)

if __name__ == '__main__':
    # import matplotlib
    # import numpy as np
    # from utils.metrics import calc_curves, calc_metrics
    # from statistics import mean
    
    
    # AVIST test code
    # gt_dir = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist/anno/'
    # prediction_dir = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist/avist_outputs/bboxes/'
    # save_dir = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist'
    # img_dir = None #'/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/data/OTB/Coke/img'
    # # print(single_eval(prediction, gt, save_dir, 0.5, img_dir, 'airplane', writeout=False))
    # # main()
    
    # out_of_view = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist/out_of_view'
    # full_occlusions = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist/full_occlusion'
    
    # vid_len = len(os.listdir(prediction_dir))
    # eao_all, robustness_50_all, robustness_75_all = [], [], []
    
    # nbins_iou=21
    # nbins_ce=51
    # succ_curve = np.zeros((vid_len, nbins_iou)) # for # of bins, default: 21
    # prec_curve = np.zeros((vid_len, nbins_ce)) # for # of bins, default: 51
    # speeds = np.zeros(vid_len)
    
    # for index, path in enumerate(os.listdir(prediction_dir)):
    #     try: 
    #         video_name = os.path.splitext(path)[0]
    #         pred_path = os.path.join(prediction_dir,path)
    #         gt_path = os.path.join(gt_dir,path)
    #         video_out_of_view = os.path.join(out_of_view,video_name+'_out_of_view.txt')      
    #         video_full_occlusions = os.path.join(full_occlusions, video_name+'_full_occlusion.txt') 
            
    #         with open(video_out_of_view) as f:
    #             lines = f.readlines()
    #             out_of_view_frames = lines[0].split(',')
            
    #         with open(video_full_occlusions) as f:
    #             lines = f.readlines()
    #             video_full_occlusions_frames = lines[0].split(',')
                
    #         prediction =  np.array(load_annotation_file_otb(pred_path))
    #         gt = np.array(load_annotation_file_otb(gt_path))

    #         # gt = np.zeros((len(prediction),5))
            
    #         # recurr_frame = 0
    #         # for curr_frame in range(len(gt)):
    #         #     if int(out_of_view_frames[curr_frame]) == 0 or int(video_full_occlusions_frames[curr_frame])==0:
    #         #         gt[curr_frame] = org_gt[recurr_frame]
    #         #         recurr_frame+=1
                            
    #         eao, robustness_50 = single_eval(prediction, gt, save_dir, 0.5, img_dir, 'vid', False)
    #         _, robustness_75 = single_eval(prediction, gt, save_dir, 0.75, img_dir, 'vid', False)
    #         eao_all.append(eao)
    #         robustness_50_all.append(robustness_50)
    #         robustness_75_all.append(robustness_75)
    #         prediction = prediction[:len(gt),:4]
    #         gt = gt[:,:4]
            
    #         prediction[:,2], prediction[:,3] = prediction[:,2]-prediction[:,0], prediction[:,3]-prediction[:,1]
    #         gt[:,2], gt[:,3] = gt[:,2]-gt[:,0], gt[:,3]-gt[:,1]
            
    #         ious, center_errors = calc_metrics(prediction, gt)
    #         succ_curve[index], prec_curve[index] = calc_curves(ious, center_errors, nbins_iou=nbins_iou, nbins_ce=nbins_ce)
    #     except FileNotFoundError as e:
    #         print(e)
    #         pass
    
    
    # gt_dir = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/got10k/OTB'
    # prediction_dir = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/got10k/OTB_outputs/bboxes'
    # save_dir = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/got10k'

    
    # vid_len = len(os.listdir(prediction_dir))
    # eao_all, robustness_50_all, robustness_75_all = [], [], []
    
    # nbins_iou=21
    # nbins_ce=51
    # succ_curve = np.zeros((vid_len, nbins_iou)) # for # of bins, default: 21
    # prec_curve = np.zeros((vid_len, nbins_ce)) # for # of bins, default: 51
    # speeds = np.zeros(vid_len)
    
    # for index, path in enumerate(os.listdir(gt_dir)):
    #     try: 
    #         video_name = os.path.splitext(path)[0]
    #         gt_path = os.path.join(gt_dir,path,'groundtruth_rect.txt')
    #         pred_path = os.path.join(prediction_dir,path+'.txt')
            
                
    #         prediction =  np.array(load_annotation_file_otb(pred_path))
    #         gt = np.array(load_annotation_file_otb(gt_path))

    #         # gt = np.zeros((len(prediction),5))
            
    #         # recurr_frame = 0
    #         # for curr_frame in range(len(gt)):
    #         #     if int(out_of_view_frames[curr_frame]) == 0 or int(video_full_occlusions_frames[curr_frame])==0:
    #         #         gt[curr_frame] = org_gt[recurr_frame]
    #         #         recurr_frame+=1
                            
    #         eao, robustness_50 = single_eval(prediction, gt, save_dir, 0.5, None, 'vid', False)
    #         _, robustness_75 = single_eval(prediction, gt, save_dir, 0.75, None, 'vid', False)
    #         eao_all.append(eao)
    #         robustness_50_all.append(robustness_50)
    #         robustness_75_all.append(robustness_75)
    #         prediction = prediction[:len(gt),:4]
    #         gt = gt[:,:4]
            
    #         prediction[:,2], prediction[:,3] = prediction[:,2]-prediction[:,0], prediction[:,3]-prediction[:,1]
    #         gt[:,2], gt[:,3] = gt[:,2]-gt[:,0], gt[:,3]-gt[:,1]
            
    #         ious, center_errors = calc_metrics(prediction, gt)
    #         succ_curve[index], prec_curve[index] = calc_curves(ious, center_errors, nbins_iou=nbins_iou, nbins_ce=nbins_ce)
    #     except FileNotFoundError as e:
    #         print(e)
    #         pass
        
    # mean_eao, mean_robustness_50_all, mean_robustness_75_all = mean(eao_all), mean(robustness_50_all), mean(robustness_75_all)
    # succ_curve = np.mean(succ_curve, axis=0)
    # prec_curve = np.mean(prec_curve, axis=0)
    # succ_score = np.mean(succ_curve)
    # prec_score = prec_curve[20]
    # succ_rate = succ_curve[nbins_iou // 2]
    # plot_curves(nbins_iou, nbins_ce, succ_curve, prec_curve, succ_score, prec_score, save_dir)   
    
    # print(mean_eao, mean_robustness_50_all, mean_robustness_75_all)

    
    save_dir = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist'
    img_dir = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist/sequences/air_show'
    prediction = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist/avist_outputs/bboxes/air_show.txt'
    gt = '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist/anno/air_show.txt'
    
    single_eval(prediction, gt, save_dir, 0.5, img_dir, 'air_show', True)