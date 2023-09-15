import pandas as pd

def convert_filenames(df_path, replace_str, replace_with_str, save_path):
    data = pd.read_csv(df_path)
    df = data.replace({'img_path': replace_str},{'img_path': replace_with_str}, regex=True)
    df.to_csv(save_path)
    
    