#!/usr/bin/python

import os, glob, subprocess, argparse, shutil
import cv2
import config as conf
import numpy as np
from tqdm import tqdm


def frame_transform(frame, ACS_case):
    '''
    This function transforms the video frames as indicated by Wang et al.
    https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Du_102_t3.pdf
    Args:
        frame: video frame to be transformed
        ACS_case: Type of transformation to apply, as indicated in ./utils/ACS_table.txt
                    I.e. 1:	φ = φ − pi/2, θ = −θ
                         2:	φ = −φ − pi/2, θ = θ
                         3:	φ = φ, θ = θ
                         5:	φ = φ + pi/2, θ = −θ
                         6:	φ = −φ + pi/2, θ = θ
                         7:	φ = φ + pi, θ = θ
                         8:	φ = −φ + pi, θ = −θ

    Returns: Transformed frame

    '''
    h, w, c = frame.shape
    new_frame = np.zeros((h,w,c), dtype='uint8')
    if ACS_case == 1:
        for x in range(w):
            for y in range(h):
                new_frame[y, x, 0] = frame[223 - y, (x + 335) % 447, 0]
                new_frame[y, x, 1] = frame[223 - y, (x + 335) % 447, 1]
                new_frame[y, x, 2] = frame[223 - y, (x + 335) % 447, 2]
        return new_frame

    elif ACS_case == 2:
        for x in range(w):
            for y in range(h):
                new_frame[y, x, 0] = frame[y, (-x + 335) % 447, 0]
                new_frame[y, x, 1] = frame[y, (-x + 335) % 447, 1]
                new_frame[y, x, 2] = frame[y, (-x + 335) % 447, 2]
        return new_frame

    elif ACS_case == 3:
        return frame

    elif ACS_case == 4:
        for x in range(w):
            for y in range(h):
                new_frame[y, x, 0] = frame[223 - y, (447 - x), 0]
                new_frame[y, x, 1] = frame[223 - y, (447 - x), 1]
                new_frame[y, x, 2] = frame[223 - y, (447 - x), 2]
        return new_frame

    elif ACS_case == 5:
        for x in range(w):
            for y in range(h):
                new_frame[y, x, 0] = frame[223 - y, (x + 111) % 447, 0]
                new_frame[y, x, 1] = frame[223 - y, (x + 111) % 447, 1]
                new_frame[y, x, 2] = frame[223 - y, (x + 111) % 447, 2]
        return new_frame

    elif ACS_case == 6:
        for x in range(w):
            for y in range(h):
                new_frame[y, x, 0] = frame[y, (-x + 111) % 447, 0]
                new_frame[y, x, 1] = frame[y, (-x + 111) % 447, 1]
                new_frame[y, x, 2] = frame[y, (-x + 111) % 447, 2]
        return new_frame

    elif ACS_case == 7:
        for x in range(w):
            for y in range(h):
                new_frame[y, x, 0] = frame[y, (x + 223) % 447, 0]
                new_frame[y, x, 1] = frame[y, (x + 223) % 447, 1]
                new_frame[y, x, 2] = frame[y, (x + 223) % 447, 2]
        return new_frame

    elif ACS_case == 8:
        for x in range(w):
            for y in range(h):
                new_frame[y, x, 0] = frame[223 - y, (-x + 223) % 447, 0]
                new_frame[y, x, 1] = frame[223 - y, (-x + 223) % 447, 1]
                new_frame[y, x, 2] = frame[223 - y, (-x + 223) % 447, 2]
        return new_frame



def main():

    if args.ACS_case == 3: # simply extract frames
        for set in glob.glob(os.path.join(conf.input['data_path'], args.subset, '*')):
            for file in glob.glob(os.path.join(set, '*')):
                destination = os.path.join(conf.input['data_path'], args.subset.replace('video', 'frames'),
                                           os.path.basename(set), '3', os.path.basename(file))
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                os.makedirs(destination)
                command = (
                            'ffmpeg -i %s -threads 10 -deinterlace -vf "fps=%d,scale=448:224" -qscale:v 1 -qmin 1 -start_number 0 %s/%s' % (
                    file, conf.input['fps'], destination, '%05d.jpg'))
                subprocess.call(command, shell=True, stdout=None)


    else: #ACS = 3 is the original φ = φ, θ = θ
        if glob.glob(os.path.join(conf.input['data_path'], args.subset.replace('video', 'frames'), '*')) == []:
            raise Exception("Need to first extract frames, then apply AVCS")

        for set in glob.glob(os.path.join(conf.input['data_path'], args.subset.replace('video', 'frames'), '*')):
            for sequence in sorted(glob.glob(os.path.join(set, str(3), '*'))):
                output_dir = sequence.replace('/3/', '/{}/'.format(str(args.ACS_case)))
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                print('ACS case {}: {}'.format(args.ACS_case, output_dir))
                for count, frame_path in enumerate(tqdm(sorted(glob.glob(os.path.join(sequence, '*'))))):
                    #print(frame_path)
                    frame = cv2.imread(frame_path)
                    transformed_frame = frame_transform(frame, args.ACS_case)
                    cv2.imwrite(os.path.join(output_dir, os.path.basename(frame_path)), transformed_frame)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and save input features')
    parser.add_argument('--subset', type=str, default='video_dev', metavar='S',
                        help='choose video_dev or video_eval')
    parser.add_argument('--ACS-case', type=int, default=3)
    args = parser.parse_args()
    main()