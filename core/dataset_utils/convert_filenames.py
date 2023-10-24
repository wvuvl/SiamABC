import pandas as pd
import os
from tqdm import tqdm

def convert_filenames(df_path, replace_str, replace_with_str, save_path):
    data = pd.read_csv(df_path)
    df = data.replace({'img_path': replace_str},{'img_path': replace_with_str}, regex=True)
    df.to_csv(save_path)
    
    
# path = '/data/zaveri/code/old_AEVT/core/dataset_utils/csvs'
# d_path = '/data/zaveri/code/old_AEVT/core/dataset_utils'

# for i in tqdm(os.listdir(path)):
#     csv_path = os.path.join(path,i)
#     new_csv_path = os.path.join(d_path,i)
#     convert_filenames(csv_path, '/data/', '/new_local_storage/', new_csv_path)


csv_path = '/new_local_storage/zaveri/code/SiamABC/core/dataset_utils/csvs/coco2017.csv'
new_csv_path = '/new_local_storage/zaveri/code/SiamABC/core/dataset_utils/coco2017.csv'
convert_filenames(csv_path, '/data/', '/new_local_storage/', new_csv_path)