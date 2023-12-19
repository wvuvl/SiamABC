import numpy as np
import os, shutil

class TrackingNetFormatPacker:
    def __init__(self, tracker_name, saving_path):
        self.tracker_name = tracker_name
        self.saving_path = saving_path

    def save_sequence_result(self, dataset_unique_id, sequence_name, predicted_bboxes, time_cost_array):
        sequence_path = os.path.join(self.saving_path, dataset_unique_id, self.tracker_name, sequence_name)
        os.makedirs(sequence_path, exist_ok=True)
        np.savetxt(os.path.join(sequence_path, f'{sequence_name}.txt'), predicted_bboxes, fmt='%.2f', delimiter=',')

    def pack_dataset_result(self, dataset_unique_id):
        archive_base_path = os.path.join(self.saving_path, f'{self.tracker_name}-{dataset_unique_id}')
        shutil.make_archive(archive_base_path, 'zip', os.path.join(self.saving_path, dataset_unique_id))
