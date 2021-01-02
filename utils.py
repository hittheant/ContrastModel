import imghdr
import os


def recursive_folder_image_paths(folder_path):
    file_paths = []
    for dirpath, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(dirpath, filename)
            if imghdr.what(file_path) is not None:
                file_paths.append(file_path)
    return file_paths
