"""
从ActivityNet的视频中提取图片帧
"""
import cv2
import os
import glob
import sys
from multiprocessing import Pool
import argparse


def dump_frames(vid_path):
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0][1:]
    out_dir_path = os.path.join(os.path.dirname(vid_path).replace(src_path, out_path), vid_name)

    fcount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    os.mkdir(out_dir_path)

    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # assert ret, '%s has %d frames, but stop at %d' % (vid_name, fcount, i)
        frame = cv2.resize(frame, (320, 256))
        cv2.imwrite('{}/{:06d}.jpg'.format(out_dir_path, count), frame)
        count = count + 1
    # print('{} done. {} frames extracted.'.format(vid_name, fcount))
    if count != fcount:
        if abs(count - fcount) > 5:
            print('%s has %d frames, but stop at %d' % (vid_path.split('/')[-1], fcount, count))
    # else:
    #     if vid_path.endswith('mkv'):
    #         print('%s has %d frames, correct' % (vid_path.split('/')[-1], fcount))
    sys.stdout.flush()

    with open(os.path.join(out_dir_path, 'n_frames'), 'w') as dst_file:
        dst_file.write(str(count))


def nonintersection(lst1, lst2):
    lst3 = [value for value in lst1 if ((value.split("/")[-1]).split(".")[0]) not in lst2]
    return lst3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("--src_dir", type=str, default='/data5/liuzhe/activitynet/v1-2')
    parser.add_argument("--out_dir", type=str, default='/data4/liuzhe/activitynet/frames')
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--resume", type=str, default='no', choices=['yes','no'], help='resume instead of overwriting')

    args = parser.parse_args()
    #
    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker

    resume = args.resume
    if not os.path.isdir(out_path):
        print("creating folder: "+out_path)
        os.makedirs(out_path)
    print("reading videos from folder: ", src_path)
    vid_list = glob.glob(src_path+'/train/*')
    print("total number of videos found: ", len(vid_list))
    if resume == 'yes':
        com_vid_list = os.listdir(out_path)
        vid_list = nonintersection(vid_list, com_vid_list)
        print("resuming from video: ", vid_list[0])
    #
    # # from tqdm import tqdm
    # #
    # # for p in tqdm(vid_list):
    # #     dump_frames(p)

    pool = Pool(num_worker)
    log = pool.map(dump_frames, vid_list)

    # for l in log:
    #     if l is not None:
    #         print(l)
    # dump_frames('/data5/liuzhe/activitynet/v1-2/train/v_uHmoFLB-PLc.mp4')
