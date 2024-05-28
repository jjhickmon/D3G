import torch
import torchvision
import csv
import os

class EquigenImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, csv_file=None, class_dicts=None):
        super(EquigenImageFolder, self).__init__(root, transform)
        assert "_output" in csv_file
        self.csv_data = {}
        if os.path.exists(csv_file):
            with open(csv_file, mode='r') as f:
                reader = csv.reader(f)
                next(reader)
                race_7_dict = {v: k for k, v in class_dicts[0].items()}
                race_4_dict = {v: k for k, v in class_dicts[1].items()}
                gender_dict = {v: k for k, v in class_dicts[2].items()}
                age_dict = {v: k for k, v in class_dicts[3].items()}
                for row in reader:
                    filename = os.path.basename(row[0]).split("_")[0]
                    race_7_label = int(race_7_dict[row[1].lower().replace('_', '/')])
                    race_4_label = int(race_4_dict[row[2].lower()])
                    gender_label = int(gender_dict[row[3].lower()])
                    age_label = int(age_dict[row[4].lower()])
                    self.csv_data[filename] = torch.tensor([race_7_label, race_4_label, gender_label, age_label])
        self.samples = [path for path in self.samples if os.path.basename(path[0]).split('.')[0] in self.csv_data.keys()]

    def __getitem__(self, index):
        original_tuple = super(EquigenImageFolder, self).__getitem__(index)
        filename = os.path.basename(self.samples[index][0]).split('.')[0]
        fairface_7_race = self.csv_data[filename][0]
        fairface_4_race = self.csv_data[filename][1]
        fairface_gender = self.csv_data[filename][2]
        fairface_age = self.csv_data[filename][3]
        return  (original_tuple[0], original_tuple[1], fairface_7_race, fairface_4_race, fairface_gender, fairface_age)