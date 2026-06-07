import os
from collections import Counter

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix
from torch.utils.data import Dataset

from .transforms import default_pc_transform
from .utils import offread_uniformed, read_from_path


class SessionDataset(Dataset):
    def __init__(self, grouped_data, beginLabel=0, transform=default_pc_transform):
        super().__init__()
        self.paths = []
        self.labels = []
        self.load_method = []
        self.transform = transform
        for label_offset, path_list in enumerate(grouped_data):
            for path, load_method in path_list:
                self.paths.append(path)
                self.labels.append(label_offset + beginLabel)
                self.load_method.append(load_method)

    def get_cat_num(self):
        return len(dict(Counter(self.labels)))

    def __getitem__(self, idx):
        item = self.paths[idx]
        if self.load_method[idx] is None:
            if isinstance(item, str):
                point_cloud = read_from_path(item)
            elif isinstance(item, torch.Tensor):
                point_cloud = item.detach().cpu().numpy()
            else:
                point_cloud = np.asarray(item, dtype=np.float32)
        else:
            point_cloud = self.load_method[idx](item)
        if self.transform is not None:
            point_cloud = self.transform(point_cloud)
        return point_cloud, self.labels[idx]

    def __len__(self):
        return len(self.paths)


class SessionMaker:
    def __init__(self):
        self.id2name = []
        self.name2id = {}
        self.data_train = []
        self.data_test = []
        self.cat_tot = 0
        self.cat_cnt_train = []
        self.cat_cnt_test = []
        self.session_cfg = []
        self.base_few_shot = 0
        self.inc_few_shot = 0
        self.memory = []
        self.base_memory = []

    def update_memory(self, exemplar):
        self.memory.append(exemplar)

    def update_base_memory(self, exemplar):
        if not any(entry["label"] == exemplar["label"] for entry in self.base_memory):
            self.base_memory.append(exemplar)

    def tot_session(self):
        return len(self.session_cfg)

    def make_session(self, session_id, update_memory=0):
        new_class_data = [[] for _ in range(self.cat_tot)]
        tmp_new_mem = []

        for label in self.session_cfg[session_id]:
            if session_id == 0:
                exemplar = {
                    "path": self.data_train[label][0][0],
                    "load_method": self.data_train[label][0][1],
                    "label": label,
                }
                self.update_base_memory(exemplar)
                new_class_data[label] = self.data_train[label]
                if self.base_few_shot > 0:
                    new_class_data[label] = new_class_data[label][: self.base_few_shot]
            else:
                new_class_data[label] = self.data_train[label]
                if self.inc_few_shot > 0:
                    new_class_data[label] = new_class_data[label][: self.inc_few_shot]
                if label >= len(self.session_cfg[0]):
                    for path, load_method in new_class_data[label][:update_memory]:
                        tmp_new_mem.append({"path": path, "load_method": load_method, "label": label})

        for exemplar in tmp_new_mem:
            self.update_memory(exemplar)

        memory_data = [[] for _ in range(self.cat_tot)]
        current_session_start = 0 if session_id == 0 else min(self.session_cfg[session_id])
        for exemplar in self.base_memory:
            if exemplar["label"] < current_session_start:
                memory_data[exemplar["label"]].append((exemplar["path"], exemplar["load_method"]))
        for exemplar in self.memory:
            if exemplar["label"] < current_session_start:
                memory_data[exemplar["label"]].append((exemplar["path"], exemplar["load_method"]))

        data_test_base = [[] for _ in range(len(self.session_cfg[0]))]
        data_test_new = [[] for _ in range(self.cat_tot - len(self.session_cfg[0]))]
        for session in range(session_id + 1):
            for label in self.session_cfg[session]:
                if label < len(self.session_cfg[0]):
                    data_test_base[label] = self.data_test[label]
                else:
                    data_test_new[label - len(self.session_cfg[0])] = self.data_test[label]

        return (
            SessionDataset(new_class_data),
            SessionDataset(memory_data),
            SessionDataset(data_test_base),
            SessionDataset(data_test_new, beginLabel=len(self.session_cfg[0])),
        )

    def get_id2name(self):
        return self.id2name

    def set_session(self, num_base_cat, num_inc_cat, base_few_shot=0, inc_few_shot=5):
        next_start = num_base_cat
        self.session_cfg = [[i for i in range(num_base_cat)]]
        while next_start < self.cat_tot:
            self.session_cfg.append([i for i in range(next_start, min(next_start + num_inc_cat, self.cat_tot))])
            next_start += num_inc_cat
        self.base_few_shot = base_few_shot
        self.inc_few_shot = inc_few_shot

    def merge_new_data(
        self,
        new_data_train,
        new_data_test,
        new_cat_cnt_train,
        new_cat_cnt_test,
        new_id2name,
        new_cat_tot,
        new_dataset_name,
    ):
        merged_cat_num = 0
        for idx in range(new_cat_tot):
            if new_cat_cnt_train[idx] == 0 and new_cat_cnt_test[idx] == 0:
                continue
            self.data_train.append(new_data_train[idx])
            self.data_test.append(new_data_test[idx])
            self.cat_cnt_train.append(new_cat_cnt_train[idx])
            self.cat_cnt_test.append(new_cat_cnt_test[idx])
            self.id2name.append(new_id2name[idx])
            self.name2id[new_id2name[idx]] = self.cat_tot
            self.cat_tot += 1
            merged_cat_num += 1
        print(f"{merged_cat_num} categories has been merged from '{new_dataset_name}'.")

    def append_dataset(self, new_dataset, new_id2name, load_method=None, split_ratio=0.8):
        new_cat_tot = len(new_id2name)
        new_cat_cnt = [0 for _ in range(new_cat_tot)]
        new_data = [[] for _ in range(new_cat_tot)]
        for path, label in new_dataset:
            new_data[label].append((path, load_method))
            new_cat_cnt[label] += 1

        new_data_train = [[] for _ in range(new_cat_tot)]
        new_data_test = [[] for _ in range(new_cat_tot)]
        new_cat_cnt_train = [0 for _ in range(new_cat_tot)]
        new_cat_cnt_test = [0 for _ in range(new_cat_tot)]
        for label, path_list in enumerate(new_data):
            num_train = int(new_cat_cnt[label] * split_ratio)
            new_data_train[label] = path_list[: num_train + 1]
            new_data_test[label] = path_list[num_train + 1 :]
            new_cat_cnt_train[label] = num_train
            new_cat_cnt_test[label] = new_cat_cnt[label] - num_train

        self.merge_new_data(
            new_data_train,
            new_data_test,
            new_cat_cnt_train,
            new_cat_cnt_test,
            new_id2name,
            new_cat_tot,
            type(new_dataset).__name__,
        )

    def append_dataset_train_test(self, new_dataset_train, new_dataset_test, new_id2name, load_method=None):
        new_cat_tot = len(new_id2name)
        new_data_train = [[] for _ in range(new_cat_tot)]
        new_data_test = [[] for _ in range(new_cat_tot)]
        new_cat_cnt_train = [0 for _ in range(new_cat_tot)]
        new_cat_cnt_test = [0 for _ in range(new_cat_tot)]

        for path, label in new_dataset_train:
            new_data_train[label].append((path, load_method))
            new_cat_cnt_train[label] += 1
        for path, label in new_dataset_test:
            new_data_test[label].append((path, load_method))
            new_cat_cnt_test[label] += 1

        self.merge_new_data(
            new_data_train,
            new_data_test,
            new_cat_cnt_train,
            new_cat_cnt_test,
            new_id2name,
            new_cat_tot,
            type(new_dataset_train).__name__,
        )

    def info(self):
        return {
            "category_num": self.cat_tot,
            "categories": {i: name for i, name in enumerate(self.id2name)},
            "train_instance_num": sum(self.cat_cnt_train),
            "train_cat_cnt": {i: (self.id2name[i], num) for i, num in enumerate(self.cat_cnt_train)},
            "test_instance_num": sum(self.cat_cnt_test),
            "test_cat_cnt": {i: (self.id2name[i], num) for i, num in enumerate(self.cat_cnt_test)},
            "session_num": len(self.session_cfg),
            "session_cfg": {i: session for i, session in enumerate(self.session_cfg)},
            "base_few_shot": self.base_few_shot,
            "inc_few_shot": self.inc_few_shot,
        }


class ShapeNetCIL(Dataset):
    id2name = [
        "table",
        "chair",
        "airplane",
        "car",
        "sofa",
        "rifle",
        "lamp",
        "vessel",
        "bench",
        "loudspeaker",
        "cabinet",
        "display",
        "bus",
        "bathtub",
        "guitar",
        "faucet",
        "clock",
        "pot",
        "telephone",
        "jar",
        "bottle",
        "laptop",
        "bookshelf",
        "knife",
        "train",
        "motorcycle",
        "ashcan",
        "file",
        "pistol",
        "piano",
        "bed",
        "stove",
        "mug",
        "bowl",
        "washer",
        "printer",
        "helmet",
        "microwave",
        "skateboard",
        "tower",
        "camera",
        "basket",
        "can",
        "pillow",
        "mailbox",
        "dishwasher",
        "rocket",
        "bag",
        "birdhouse",
        "earphone",
        "microphone",
        "remote control",
        "cap",
        "cellular telephone",
        "computer keyboard",
    ]
    cat_labels = {
        "04379243": 0,
        "03001627": 1,
        "02691156": 2,
        "02958343": 3,
        "04256520": 4,
        "04090263": 5,
        "03636649": 6,
        "04530566": 7,
        "02828884": 8,
        "03691459": 9,
        "02933112": 10,
        "03211117": 11,
        "02924116": 12,
        "02808440": 13,
        "03467517": 14,
        "03325088": 15,
        "03046257": 16,
        "03991062": 17,
        "04401088": 18,
        "03593526": 19,
        "02876657": 20,
        "03642806": 21,
        "02871439": 22,
        "03624134": 23,
        "04468005": 24,
        "03790512": 25,
        "02747177": 26,
        "03337140": 27,
        "03948459": 28,
        "03928116": 29,
        "02818832": 30,
        "04330267": 31,
        "03797390": 32,
        "02880940": 33,
        "04554684": 34,
        "04004475": 35,
        "03513137": 36,
        "03761084": 37,
        "04225987": 38,
        "04460130": 39,
        "02942699": 40,
        "02801938": 41,
        "02946921": 42,
        "03938244": 43,
        "03710193": 44,
        "03207941": 45,
        "04099429": 46,
        "02773838": 47,
        "02843684": 48,
        "03261776": 49,
        "03759954": 50,
        "04074963": 51,
        "02954340": 52,
        "02992529": 53,
        "03085013": 54,
    }

    def __init__(self, root="./data/shapenet", partition="train", banlist=None, whole=False):
        assert partition in ["train", "test"]
        banlist = banlist or []
        self.data_root = os.path.join(root, "ShapeNet-55")
        self.pc_path = os.path.join(root, "shapenet_pc")
        self.subset = partition
        self.whole = whole
        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")
        test_data_list_file = os.path.join(self.data_root, "test.txt")

        with open(self.data_list_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
        if self.whole:
            with open(test_data_list_file, "r", encoding="utf-8") as fin:
                lines = fin.readlines() + lines

        check_list = [
            "03001627-udf068a6b",
            "03001627-u6028f63e",
            "03001627-uca24feec",
            "04379243-",
            "02747177-",
            "03001627-u481ebf18",
            "03001627-u45c7b89f",
            "03001627-ub5d972a1",
            "03001627-u1e22cc04",
            "03001627-ue639c33f",
        ]
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split("-")[0]
            model_id = line.split("-")[1].split(".")[0]
            if ShapeNetCIL.id2name[ShapeNetCIL.cat_labels[taxonomy_id]] in banlist:
                continue
            if taxonomy_id + "-" + model_id not in check_list:
                self.file_list.append(
                    {
                        "taxonomy_id": taxonomy_id,
                        "model_id": model_id,
                        "file_path": line,
                    }
                )

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        path = os.path.join(self.pc_path, sample["file_path"])
        return path, ShapeNetCIL.cat_labels[sample["taxonomy_id"]]

    def __len__(self):
        return len(self.file_list)


class ModelNet40AlignCIL(Dataset):
    cats = {
        "airplane": 0,
        "bathtub": 1,
        "bed": 2,
        "bench": 3,
        "bookshelf": 4,
        "bottle": 5,
        "bowl": 6,
        "car": 7,
        "chair": 8,
        "cone": 9,
        "cup": 10,
        "curtain": 11,
        "desk": 12,
        "door": 13,
        "dresser": 14,
        "flower_pot": 15,
        "glass_box": 16,
        "guitar": 17,
        "keyboard": 18,
        "lamp": 19,
        "laptop": 20,
        "mantel": 21,
        "monitor": 22,
        "night_stand": 23,
        "person": 24,
        "piano": 25,
        "plant": 26,
        "radio": 27,
        "range_hood": 28,
        "sink": 29,
        "sofa": 30,
        "stairs": 31,
        "stool": 32,
        "table": 33,
        "tent": 34,
        "toilet": 35,
        "tv_stand": 36,
        "vase": 37,
        "wardrobe": 38,
        "xbox": 39,
    }
    id2name = list(cats.keys())

    def __init__(self, root="./data/modelnet", pt_num=10000, partition="train", banlist=None):
        assert partition in ("test", "train")
        super().__init__()
        self.root = root
        self.partition = partition
        self.pt_num = pt_num
        self._load_data(banlist or [])

    def _load_data(self, banlist):
        self.paths = []
        self.labels = []
        for cat in os.listdir(self.root):
            if cat in banlist:
                continue
            cat_path = os.path.join(self.root, cat, self.partition)
            for case in os.listdir(cat_path):
                if case.endswith(".off"):
                    self.paths.append(os.path.join(cat_path, case))
                    self.labels.append(ModelNet40AlignCIL.cats[cat])

    def get_load_method(self):
        def load(path, pt_num=self.pt_num):
            points = torch.tensor(offread_uniformed(path, sampled_pt_num=pt_num), dtype=torch.float32)
            rota1 = axis_angle_to_matrix(torch.tensor([0.5 * np.pi, 0, 0]))
            rota2 = axis_angle_to_matrix(torch.tensor([0, -0.5 * np.pi, 0]))
            return (points @ rota1 @ rota2).numpy()

        return load

    def __getitem__(self, index):
        return self.paths[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class CO3DCIL_S2C(Dataset):
    cats = {
        "apple": 0,
        "backpack": 1,
        "ball": 2,
        "banana": 3,
        "baseballbat": 4,
        "baseballglove": 5,
        "bench": 6,
        "bicycle": 7,
        "book": 8,
        "bottle": 9,
        "bowl": 10,
        "broccoli": 11,
        "cake": 12,
        "car": 13,
        "carrot": 14,
        "cellphone": 15,
        "chair": 16,
        "couch": 17,
        "cup": 18,
        "donut": 19,
        "frisbee": 20,
        "hairdryer": 21,
        "handbag": 22,
        "hotdog": 23,
        "hydrant": 24,
        "keyboard": 25,
        "kite": 26,
        "laptop": 27,
        "microwave": 28,
        "motorcycle": 29,
        "mouse": 30,
        "orange": 31,
        "parkingmeter": 32,
        "pizza": 33,
        "plant": 34,
        "remote": 35,
        "sandwich": 36,
        "skateboard": 37,
        "stopsign": 38,
        "suitcase": 39,
        "teddybear": 40,
        "toaster": 41,
        "toilet": 42,
        "toybus": 43,
        "toyplane": 44,
        "toytruck": 45,
        "tv": 46,
        "umbrella": 47,
        "vase": 48,
        "wineglass": 49,
    }
    id2name = list(cats.keys())

    def __init__(self, root="./data/co3d", pt_num=10000, partition="train", banlist=None):
        assert partition in ("test", "train")
        super().__init__()
        self.root = root
        self.partition = partition
        self.pt_num = pt_num
        self._load_data(banlist or [])

    def _load_data(self, banlist):
        self.paths = []
        self.labels = []
        for cat in os.listdir(self.root):
            if cat in banlist:
                continue
            cat_path = os.path.join(self.root, cat, self.partition)
            for case in os.listdir(cat_path):
                if case.endswith(".ply"):
                    self.paths.append(os.path.join(cat_path, case))
                    self.labels.append(CO3DCIL_S2C.cats[cat])

    def __getitem__(self, index):
        return self.paths[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class CO3DCIL(Dataset):
    cats = {
        "backpack": 0,
        "handbag": 1,
        "carrot": 2,
        "teddybear": 3,
        "bowl": 4,
        "chair": 5,
        "vase": 6,
        "book": 7,
        "keyboard": 8,
        "plant": 9,
        "ball": 10,
        "hairdryer": 11,
        "umbrella": 12,
        "laptop": 13,
        "orange": 14,
        "toytruck": 15,
        "suitcase": 16,
        "wineglass": 17,
        "mouse": 18,
        "cellphone": 19,
        "broccoli": 20,
        "remote": 21,
        "apple": 22,
        "cake": 23,
        "toilet": 24,
        "bicycle": 25,
        "hydrant": 26,
        "toaster": 27,
        "bottle": 28,
        "motorcycle": 29,
        "sandwich": 30,
        "bench": 31,
        "toyplane": 32,
        "couch": 33,
        "car": 34,
        "banana": 35,
        "donut": 36,
        "stopsign": 37,
        "cup": 38,
        "kite": 39,
        "toybus": 40,
        "pizza": 41,
        "frisbee": 42,
        "baseballglove": 43,
        "skateboard": 44,
        "baseballbat": 45,
        "hotdog": 46,
        "microwave": 47,
        "tv": 48,
        "parkingmeter": 49,
    }
    id2name = list(cats.keys())

    def __init__(self, root="./data/co3d", pt_num=10000, partition="train", banlist=None):
        assert partition in ("test", "train")
        super().__init__()
        self.root = root
        self.partition = partition
        self.pt_num = pt_num
        self._load_data(banlist or [])

    def _load_data(self, banlist):
        self.paths = []
        self.labels = []
        for cat in os.listdir(self.root):
            if cat in banlist:
                continue
            cat_path = os.path.join(self.root, cat, self.partition)
            for case in os.listdir(cat_path):
                if case.endswith(".ply"):
                    self.paths.append(os.path.join(cat_path, case))
                    self.labels.append(CO3DCIL.cats[cat])

    def __getitem__(self, index):
        return self.paths[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class ScanObjNN(Dataset):
    cats = {
        "bag": 0,
        "bin": 1,
        "box": 2,
        "cabinet": 3,
        "chair": 4,
        "desk": 5,
        "display": 6,
        "door": 7,
        "shelf": 8,
        "table": 9,
        "bed": 10,
        "pillow": 11,
        "sink": 12,
        "sofa": 13,
        "toilet": 14,
    }
    id2name = list(cats.keys())

    def __init__(self, root="./data/scanobjnn", pt_num=10000, partition="train", banlist=None):
        assert partition in ("test", "train")
        super().__init__()
        self.root = root
        self.partition = partition
        self.pt_num = pt_num
        self._load_data(banlist or [])

    def _load_data(self, banlist):
        self.paths = []
        self.labels = []
        for cat in os.listdir(self.root):
            if cat in banlist:
                continue
            cat_path = os.path.join(self.root, cat, self.partition)
            for case in os.listdir(cat_path):
                if case.endswith(".ply"):
                    self.paths.append(os.path.join(cat_path, case))
                    self.labels.append(ScanObjNN.cats[cat])

    def __getitem__(self, index):
        return self.paths[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def pc_normalize_np(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    radius = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if radius > 0:
        pc = pc / radius
    return pc


def normalize_pc(pc):
    pc = pc - np.mean(pc, axis=0)
    max_norm = np.max(np.linalg.norm(pc, axis=1))
    if max_norm < 1e-6:
        return np.zeros_like(pc)
    return pc / max_norm
