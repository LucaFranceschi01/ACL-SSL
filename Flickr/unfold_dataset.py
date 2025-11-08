import pandas as pd
import os
import sys
import shutil

# usage: [lukovsky@fedora Flickr]$ python unfold_dataset.py 0
# argument 1 for dry-run

def list_data():
    data_index = {}
    for idx in os.listdir('Dataset/Data'):
        data_index[idx] = os.listdir('Dataset/Data/' + idx)

    for idx, data in data_index.items():
        data_index[idx] = set([int(s.split('.')[0]) for s in data])
    
    return data_index

def main():

    DRY_RUN = bool(int(sys.argv[1])) # param true false

    metadata_flickr_test = pd.read_csv('metadata/flickr_test.csv', header=None)
    metadata_flickr_test[0] = metadata_flickr_test[0].astype('int64')

    data_index = list_data()
    
    os.makedirs('test/audio', exist_ok=True)
    os.makedirs('test/frames', exist_ok=True)

    for test_datapoint in metadata_flickr_test[0]:
        for idx, filenames in data_index.items():
            if test_datapoint in filenames:
                
                # one of the worst things ive done in my life but whatever
                if not DRY_RUN:
                    os.rename('Dataset/Data/'+idx+'/'+str(test_datapoint)+'.wav', 'test/audio/'+str(test_datapoint)+'.wav')
                    os.rename('Dataset/Data/'+idx+'/'+str(test_datapoint)+'.jpg', 'test/frames/'+str(test_datapoint)+'.jpg')

    data_index = list_data()

    os.makedirs('audio', exist_ok=True)
    os.makedirs('frames', exist_ok=True)

    for idx, filenames in data_index.items():
        for filename in filenames:
                
            if not DRY_RUN:
                # both files might not exist
                if os.path.isfile('Dataset/Data/'+idx+'/'+str(filename)+'.wav') and os.path.isfile('Dataset/Data/'+idx+'/'+str(filename)+'.jpg'):
                    os.rename('Dataset/Data/'+idx+'/'+str(filename)+'.wav', 'audio/'+str(filename)+'.wav')
                    os.rename('Dataset/Data/'+idx+'/'+str(filename)+'.jpg', 'frames/'+str(filename)+'.jpg')

    if not DRY_RUN:
        shutil.move('Dataset/Annotations', 'test/Annotations')
        shutil.rmtree('Dataset')


if (__name__ == '__main__'):
    main()