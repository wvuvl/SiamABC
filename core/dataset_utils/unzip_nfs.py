import os
from tqdm import tqdm
import zipfile
import argparse
import shutil


def main(dir, dest):

    for zip_file in tqdm(os.listdir(dir)):
        if (zip_file.endswith('.zip')):

            frame_folder = os.path.join(dest, os.path.splitext(zip_file)[0])

            try:
                with zipfile.ZipFile(os.path.join(dir, zip_file)) as zip_ref:
                    
                    # create frame folder if does not exist already
                    if (os.path.exists(frame_folder)):

                        # Check if there is the same number of files within the folder
                        same_number_files = len(zip_ref.infolist()) == len(os.listdir(frame_folder))
                                                    
                        if not same_number_files:
                            shutil.rmtree(frame_folder)
                            print("overwriting", frame_folder, "due to different number of file in the folder.")
                            os.makedirs(frame_folder)

                    # if frame folder does not exist, jsut create it
                    else:	
                        same_number_files = False				
                        os.makedirs(frame_folder)

                    # extract zip if necessary
                    if not same_number_files:
                        zip_ref.extractall(os.path.join(frame_folder))

                    # check that all the files were extracted
                    same_number_files = len(zip_ref.infolist()) == len(os.listdir(frame_folder))
                    if (not same_number_files):
                        print("Warning:", frame_folder, "was not well extracted")

            except zipfile.BadZipFile:
                print("Error: the zip file", zip_file, "is corrupted, please delete it and download it again.")
                
main('/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/SOTA_datasets/NFS/zips',
     '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/SOTA_datasets/NFS/zips/NFS')