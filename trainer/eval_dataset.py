from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import pickle
import os

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
    return file_paths  # Self-explanatory.

class load_IUB:
def __init__(self, pickle_path, number_test_set, data_path):
    with open(pickle_path, "rb") as fp:  # Unpickling
            [self.IUB_cosine_data_path, self.IUB_cosine_mos] = pickle.load(fp)
            [self.IUB_voices_data_path, self.IUB_voices_mos] = pickle.load(fp)
        
        self.data_path = data_path
        self.IUB_cosine_data_dict = dict()
        for file in self.IUB_cosine_data_path[0:number_test_set]:
            # Join the root directory with the relative file path
            file = os.path.join(self.data_path, file.lstrip('/'))
            noisy, fs = torchaudio.load(file)
            self.IUB_cosine_data_dict[file] = noisy
        self.IUB_cosine_mos = self.IUB_cosine_mos[0:number_test_set]  
        
        self.IUB_voices_data_dict = dict()
        for file in self.IUB_voices_data_path[0:number_test_set]:
            # Join the root directory with the relative file path
            file = os.path.join(self.data_path, file.lstrip('/'))
            noisy, fs = torchaudio.load(file)
            self.IUB_voices_data_dict[file] = noisy
        self.IUB_voices_mos = self.IUB_voices_mos[0:number_test_set] 
        
    
class load_Tencent:
    def __init__(self, pickle_path, number_test_set, data_path):
        with open(pickle_path, "rb") as fp:  # Unpickling
            [Tencent_woR_data_path, Tencent_woR_mos] = pickle.load(fp)
            [Tencent_wR_data_path, Tencent_wR_mos] = pickle.load(fp)
        
        self.Tencent_woR_data_dict = dict()
        self.Tencent_woR_data_path = []
        self.Tencent_woR_mos = []
        self.data_path = data_path
        n=0     
        for i, file in enumerate(Tencent_woR_data_path):
            # Join the root directory with the relative file path
            file = os.path.join(self.data_path, file.lstrip('/'))
            noisy, fs = torchaudio.load(file)
            if fs == 16000:
                self.Tencent_woR_data_dict[file] = noisy
                self.Tencent_woR_data_path.append(file)
                self.Tencent_woR_mos.append(Tencent_woR_mos[i])
                n += 1
            if n == number_test_set:
                break       
        
        self.Tencent_wR_data_dict = dict()
        self.Tencent_wR_data_path = []
        self.Tencent_wR_mos = []
        n=0     
        for i, file in enumerate(Tencent_wR_data_path):
            # Join the root directory with the relative file path
            file = os.path.join(self.data_path, file.lstrip('/'))
            noisy, fs = torchaudio.load(file)
            if fs == 16000 and Tencent_wR_mos[i]>1.1:
                self.Tencent_wR_data_dict[file] = noisy
                self.Tencent_wR_data_path.append(file)
                self.Tencent_wR_mos.append(Tencent_wR_mos[i])
                n += 1
            if n == number_test_set:
                break   
        
class load_DNS1:
    def __init__(self, dir_path):   
        DNS1_Real_list = get_filepaths(dir_path+"/real")
        DNS1_Noreverb_list = get_filepaths(dir_path+"/noreverb")
        DNS1_Reverb_list = get_filepaths(dir_path+"/reverb")
        
        self.DNS1_Real_dict = dict()
        for file in DNS1_Real_list:
            noisy, fs = torchaudio.load(file)
            self.DNS1_Real_dict[file] = noisy
            
        self.DNS1_Noreverb_dict = dict()
        for file in DNS1_Noreverb_list:
            noisy, fs = torchaudio.load(file)
            self.DNS1_Noreverb_dict[file] = noisy
            
        self.DNS1_Reverb_dict = dict()
        for file in DNS1_Reverb_list:
            noisy, fs = torchaudio.load(file)
            self.DNS1_Reverb_dict[file] = noisy
        
class load_DNS3: 
    def __init__(self, dir_path):
        DNS3_list = get_filepaths(dir_path)     
        self.DNS3_nonenglish_synthetic_dict = dict()
        self.DNS3_stationary_dict = dict()
        self.DNS3_ms_realrec_nonenglish_dict = dict()
        self.DNS3_ms_realrec_dict = dict()
        
        for file in DNS3_list:     
            noisy, fs = torchaudio.load(file)
            if 'ms_realrec_nonenglish' in file:
                self.DNS3_ms_realrec_nonenglish_dict[file] = noisy
            elif 'nonenglish' in file:
                self.DNS3_nonenglish_synthetic_dict[file] = noisy
            elif 'stationary_english' in file:
                self.DNS3_stationary_dict[file] = noisy
            else:
                self.DNS3_ms_realrec_dict[file] = noisy
            
class load_VCTK_testSet:  
    def __init__(self, pickle_path, data_path):
        with open(pickle_path, "rb") as fp:  # Unpickling
            self.vctk_Noisy_list = pickle.load(fp)
            self.SNR_list = pickle.load(fp)
            self.PESQ_list = pickle.load(fp)
            test_DNSMOSp835 = pickle.load(fp)
            self.STOI_list = pickle.load(fp)
            self.data_path = data_path
            
        self.sig = [i['SIG'] for i in test_DNSMOSp835]
        self.bak = [i['BAK'] for i in test_DNSMOSp835]
        self.ovr = [i['OVRL'] for i in test_DNSMOSp835] 
        
        self.VCTK_data_dict = dict()
        for file in self.vctk_Noisy_list:
            # Join the root directory with the relative file path
            file = os.path.join(self.data_path, file.lstrip('/'))
            noisy, fs = torchaudio.load(file)
            self.VCTK_data_dict[file] = noisy 
            
class load_VCTK_validSet:  
    def __init__(self, pickle_path):    
        with open(pickle_path, "rb") as fp:  # Unpickling
            self.valid_list = pickle.load(fp)
                             
        self.VCTK_data_dict = dict()
        for file in self.valid_list:
            noisy, fs = torchaudio.load(file)
            self.VCTK_data_dict[file] = noisy 
            
