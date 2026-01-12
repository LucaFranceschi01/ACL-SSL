import pandas as pd
import os
import sys
import shutil

def list_data():
    '''
    list data that is both in audio and videos
    '''
    audios = set([int(s.split('.')[0]) for s in os.listdir('flickr-test-plus-silent/audio')])
    frames = set([int(s.split('.')[0]) for s in os.listdir('flickr-test-plus-silent/frames')])
    
    return audios.intersection(frames)

def main():

    metadata_flickr_test = pd.read_csv('metadata/flickr_test_plus_silent.csv')
    # turns out audio col is a complete subset of video col
    metadata_flickr_test['video'] = metadata_flickr_test['video'].astype('int64')

    data_index = list_data()
    
    os.makedirs('test/extend_audio', exist_ok=True)
    os.makedirs('test/extend_frames', exist_ok=True)

    for test_datapoint in metadata_flickr_test['video']:
        if test_datapoint in data_index:
            os.rename('flickr-test-plus-silent/audio/'+str(test_datapoint)+'.wav', 'test/extend_audio/'+str(test_datapoint)+'.wav')
            os.rename('flickr-test-plus-silent/frames/'+str(test_datapoint)+'.jpg', 'test/extend_frames/'+str(test_datapoint)+'.jpg')

    data_index = list_data()

    os.makedirs('extend_audio', exist_ok=True)
    os.makedirs('extend_frames', exist_ok=True)

    for filename in data_index:
        os.rename('flickr-test-plus-silent/audio/'+str(filename)+'.wav', 'extend_audio/'+str(filename)+'.wav')
        os.rename('flickr-test-plus-silent/frames/'+str(filename)+'.jpg', 'extend_frames/'+str(filename)+'.jpg')

    # shutil.move('flickr-test-plus-silent/Annotations/*', 'test/Annotations')
    # already there bc of base flickr dataset
    shutil.rmtree('flickr-test-plus-silent')


if (__name__ == '__main__'):
    main()