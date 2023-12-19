import os
import pandas as pd
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import json

def printBB(TrackingNet_dir, frames_folder, BB_file):

    ArrayBB = np.loadtxt(BB_file, delimiter=",")  

    frames_list=[os.path.join(TrackingNet_dir, frame) for frame in os.listdir(frames_folder) if frame.endswith(".jpg") ]

    if ( not len(ArrayBB) == len(frames_list)):
        #print("Not the same number of frames and annotation!" ) 
        if (np.ndim(ArrayBB) == 1):
            tmp = ArrayBB
            del ArrayBB
            ArrayBB = [[]]
            ArrayBB[0] = tmp


    return ArrayBB





def main(output_dir="TrackingNet", save_file="trackingnet.csv", chunks=[]):

    
    sequence_id = 0
    data = []
    json_dict = {}
    for chunk_folder in chunks:
        chunk_folder = chunk_folder.upper()


        list_sequences = os.listdir(os.path.join(output_dir, chunk_folder, "frames"))

        for seq_ID in tqdm(list_sequences, desc=chunk_folder):

            frames_folder = os.path.join(output_dir, chunk_folder, "frames", seq_ID)
            BB_file = os.path.join(output_dir, chunk_folder, "anno", seq_ID + ".txt")
            ArrayBB = printBB(output_dir, frames_folder=frames_folder, BB_file=BB_file)

            
           
            image_files = []
            gt_rect = []
            for i in range(len(ArrayBB)):

                frame_file = str(i)+".jpg"

                imgs_file = os.path.join(frames_folder, frame_file)
                image_files.append(os.path.join(seq_ID, frame_file))
                
                img = Image.open(imgs_file)
                image_width,image_height  = img.size
        
                x = int(ArrayBB[i][0])
                y = int(ArrayBB[i][1])
                w = int(ArrayBB[i][2])
                h = int(ArrayBB[i][3])
                
                
                bbox_exist = int((w > 0) & (h > 0))
                bbox_border = int(x<=0 or y<=0 or (x+w)>=image_width-1 or (y+h)>=image_height-1)
                x = x if x>0 else 0
                y = y if y>0 else 0
                
                data.append([str(sequence_id), seq_ID, i, imgs_file, [x,y,w,h], [image_width, image_height], 'TrackingNet',bbox_exist, bbox_border])
                gt_rect.append([x,y,w,h])
            
            json_dict[seq_ID] = {}
            json_dict[seq_ID]['video_dir'] = seq_ID
            json_dict[seq_ID]['img_names'] = image_files
            json_dict[seq_ID]['init_rect'] = gt_rect[0]
            json_dict[seq_ID]['gt_rect'] = gt_rect
            
            sequence_id+=1
            # sanity save
            # df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])

            
    # df = pd.DataFrame(data, columns=["sequence_id","track_id","frame_index","img_path","bbox","frame_shape","dataset","presence","near_corner"])
    # df.to_csv(save_file)
    with open(f"TrackingNet_{chunk_folder}.json", "w") as outfile: 
        json.dump(json_dict, outfile)



path = '/new_local_storage/zaveri/SOTA_Tracking_datasets/TrackingNet'

if __name__ == "__main__": 
    p = argparse.ArgumentParser(description='Download the frames for TrackingNet')
    p.add_argument('--output_dir', type=str, default=path,
        help='Main TrackingNet folder.')
    p.add_argument('--save_file', type=str, default=False,
        help='Folder where to store the csv file.')
    p.add_argument('--chunk', type=str, default="ALL",
        help='List of chunks to elaborate [ALL / 4 / 1,2,5].')

    args = p.parse_args()
    
    if 'ALL' in args.chunk.upper():
        args.chunk = ["TRAIN_"+str(c) for c in range(12)]  
    else:
        args.chunk = ["TRAIN_"+args.chunk]

    
    print("Draw Bounding Boxes on the annotated frames for the following chunks")
    print("CHUNKS:", args.chunk)

    main(output_dir=args.output_dir, 
        save_file=args.save_file , 
        chunks=args.chunk)


