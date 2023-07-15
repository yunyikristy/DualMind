# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import io
import os
import pickle
import zipfile

import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_zip_path(img_or_path):
    """judge if this is a zip path"""
    return '.zip@' in img_or_path


class ZipReader(object):
    """A class to read zipped files"""
    zip_bank = dict()
    zip_list = []
    zip_size = 1000

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path not in zip_bank:
            zfile = zipfile.ZipFile(path, 'r')
            ZipReader.zip_list.append(path)
            zip_bank[path] = zfile
            if len(ZipReader.zip_list) > ZipReader.zip_size:
                index = ZipReader.zip_list[0]
                ZipReader.zip_list = ZipReader.zip_list[-ZipReader.zip_size:]
                zf = zip_bank.pop(index)
                zf.close()
        else:
            ZipReader.zip_list.remove(path)
            ZipReader.zip_list.append(path)
        return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        pos_at = path.index('@')
        assert pos_at != -1, "character '@' is not found from the given path '%s'" % path

        zip_path = path[0: pos_at]
        folder_path = path[pos_at + 1:]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path

    @staticmethod
    def list_folder(path):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        folder_list = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and \
                    len(os.path.splitext(file_foler_name)[-1]) == 0 and \
                    file_foler_name != folder_path:
                if len(folder_path) == 0:
                    folder_list.append(file_foler_name)
                else:
                    folder_list.append(file_foler_name[len(folder_path) + 1:])

        return folder_list

    @staticmethod
    def list_files(path, extension=None):
        if extension is None:
            extension = ['.*']
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and \
                    str.lower(os.path.splitext(file_foler_name)[-1]) in extension:
                if len(folder_path) == 0:
                    file_lists.append(file_foler_name)
                else:
                    file_lists.append(file_foler_name[len(folder_path) + 1:])

        return file_lists

    @staticmethod
    def read(zip_path, path_img):
        # zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data

    @staticmethod
    def imread(zip_path, path_img):
        # zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        try:
            im = Image.open(io.BytesIO(data))
        except:
            print("ERROR IMG LOADED: ", path_img)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im


class PickleReader(object):
    """A class to read pickle files"""
    pickle_bank = dict()
    pickle_list = []
    pickle_size = 200000

    def __init__(self):
        super(PickleReader, self).__init__()

    @staticmethod
    def get_picklefile(path):
        pickle_bank = PickleReader.pickle_bank
        if path not in pickle_bank:
            pfile = open(path, 'rb')
            p_data = pickle.load(pfile)
            pfile.close()
            PickleReader.pickle_list.append(path)
            pickle_bank[path] = p_data
            if len(PickleReader.pickle_list) > PickleReader.pickle_size:
                index = PickleReader.pickle_list[0]
                PickleReader.pickle_list = PickleReader.pickle_list[-PickleReader.pickle_size:]
                pickle_bank.pop(index)
                # zf.close()
        else:
            PickleReader.pickle_list.remove(path)
            PickleReader.pickle_list.append(path)
        return pickle_bank[path]

    @staticmethod
    def read(pickle_path):
        # zip_path, path_img = ZipReader.split_zip_style_path(path)
        data = PickleReader.get_picklefile(pickle_path)
        # print(data)
        # data = zfile.read(path_img)
        return data
