import pickle
import os
import joblib

def save_object(file_path, object):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    # with open(file_path, 'wb') as file_object:
    #     pickle.dump(object, file_object)
    joblib.dump(object, file_path)

def load_object(file_path):
    with open(file_path, 'rb') as file_object:
        return pickle.load(file_object)