'''
Docstring for VGGSS.unfold_large_dataset

Meant to modify total_video_frames into frames, by taking the center frame from each video and copying it to frames with the correct name.
'''
import os, shutil

if __name__ == '__main__':

    src_data_dir = '/projects/imva/vggsound/train/total_video_frames'
    dst_data_dir = '/home/lfranceschi/repos/ACL-SSL/vggsound/frames'

    os.makedirs(dst_data_dir, exist_ok=True)

    video_names = os.listdir(src_data_dir)

    for i, v_name in enumerate(video_names):
        frames = os.listdir(os.path.join(src_data_dir, v_name))

        if len(frames) == 0:
            continue

        center_frame = frames[round(len(frames)/2)]

        if i%100 == 0:
            print(f'Moving file {i}/{len(video_names)}')

        shutil.copy(os.path.join(src_data_dir, v_name, center_frame), os.path.join(dst_data_dir, v_name + '.jpg'))
