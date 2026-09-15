import os
import json
import torch
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
import csv
from dataset import transforms as T

def add_pair(samples, seen_pairs, im0, msk0, im1, msk1, pa_id, nod_id, ta, tb, dt, age_val, gen, pair_label):
    """Add a pair to samples list if not already present (deduplication via seen_pairs set)."""
    key = (pa_id, nod_id, ta, tb)
    if key in seen_pairs:
        print(f"  Skipping duplicate pair: {key}")
        return
    seen_pairs.add(key)
    samples.append({
        "im0": im0, "msk0": msk0,
        "im1": im1, "msk1": msk1,
        "PatientId": pa_id,
        "NoduleId": nod_id,
        "t0": ta,
        "t1": tb,
        "dt": float(dt),
        "age0": age_val,
        "gender": gen,
        "pair": pair_label
    })

def gradient(volume):

    gx = ndimage.sobel(volume, axis=0)
    gy = ndimage.sobel(volume, axis=1)
    gz = ndimage.sobel(volume, axis=2)

    return gx, gy, gz
 
def tenengrad(volume):

    gx, gy, gz = gradient(volume)
    tenengrad = np.mean(gx**2 + gy**2 + gz**2)

    return tenengrad

def tenengrad_per_axis(volume):
    gx = ndimage.sobel(volume, axis=0)
    gy = ndimage.sobel(volume, axis=1)
    gz = ndimage.sobel(volume, axis=2)
    return np.mean(gx**2), np.mean(gy**2), np.mean(gz**2)
 
class NGPPairDataset(Dataset):
    def __init__(self, root_dir, mode, dim, diff, num_folds=5, fold=0):
        """
        Args:
            root_dir (str): The root directory where images/json are stored.
            mode (str): 'train' or 'valid' / 'test'.
            dim (int or tuple): The target spatial dimension (e.g., 64).
            diff (bool): Whether to output the differences or raw images.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.dim = dim
        self.diff = diff

        print(f"Diff is set to {self.diff} for NGPPairDataset")

        # 1. Parse Data List using original logic
        if self.mode == 'train':
            data_list = get_data_list(root_dir, "training")
            tra_data, val_data = split_data_list(data_list, num_folds, fold=fold)
            self.data_list = tra_data 

        elif self.mode == 'valid':
            data_list = get_data_list(root_dir, "training")
            tra_data, val_data = split_data_list(data_list, num_folds, fold=fold)
            self.data_list = val_data
        else:
            self.data_list = get_data_list(root_dir, "test")

        self.samples = []  
        seen_pairs = set()
        print(f"mode is {self.mode} and number of subjects in data_list is {len(self.data_list)}")

        # 1. Parse NGP Triplets into Pairs
        # Each item in data_list is a subject with 3 time points
        for subject in self.data_list:
            # Extract paths
            # series is a list of 3 dicts: [{'image':.., 'label':..}, {T2}, {T3}]
            series = subject["series"]
            info = subject["info"]

            age = float(info["Age"])     # Baseline age at T1
            gender = float(info["Gender"])

            #get subject id, nodule id, and time points for better tracking
            subject_id = info["PatientId"]
            nodule_id = info["NoduleId"]
            t0 = info["t0"]
            t1 = info["t1"]
            t2 = info["t2"]
            
            # Paths relative to root_dir
            t1_img = os.path.join(root_dir, series[0]["image"].replace(".gz", ""))
            t1_msk = os.path.join(root_dir, series[0]["label"].replace(".gz", ""))
            
            t2_img = os.path.join(root_dir, series[1]["image"].replace(".gz", ""))
            t2_msk = os.path.join(root_dir, series[1]["label"].replace(".gz", ""))
            
            t3_img = os.path.join(root_dir, series[2]["image"].replace(".gz", ""))
            t3_msk = os.path.join(root_dir, series[2]["label"].replace(".gz", ""))

            # Time intervals (integers)
            m1 = int(info["Months1"]) # Interval T1 -> T2
            m2 = int(info["Months2"]) # Interval T2 -> T3

            
            # --- Create 3 Pairs per Subject ---
            if self.mode == 'train': # to match NGP test fold (t1,t2 predict t3 only)
                add_pair(self.samples, seen_pairs,
                         t1_img, t1_msk, t2_img, t2_msk,
                         subject_id, nodule_id, t0, t1, m1, age, gender, 12)

                # Pair 3: T1 -> T3
                add_pair(self.samples, seen_pairs,
                        t1_img, t1_msk, t3_img, t3_msk,
                        subject_id, nodule_id, t0, t2, m1 + m2, age, gender, 13)

            #Pair 2: T2 -> T3
            add_pair(self.samples, seen_pairs,
                     t2_img, t2_msk, t3_img, t3_msk,
                     subject_id, nodule_id, t1, t2, m2, int(age + m1 / 12.0), gender, 23)

        
        print(f"[{mode}] Parsed {len(self.data_list)} triplets into {len(self.samples)} pairs.")

        # 2. Setup Preprocessing (Using your original custom Transforms!)
        SCOPE = (-1200, 600)
        RANGE = (-1.0, 1.0) # Explicitly forcing [-1, 1] range here
        SHAPE = [self.dim] * 3 if isinstance(self.dim, int) else self.dim

        if mode == 'train':
            self.transforms = T.Compose([
                T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
                T.RandomCrop(rand_crop=False, crop_size=SHAPE),
                T.ScaleIntensity(scope=SCOPE, range=RANGE),
                T.RandomFilp(prob=0.2, axes=(0, 1, 2)),
                T.RandomRot90(prob=0.2, axes=(0, 1, 2)),
                T.RandomScaleIntensity(prob=0.1, factor=0.1),
                T.RandomShiftIntensity(prob=0.1, offset=0.1*(RANGE[1]-RANGE[0])),
                T.AddChannel(img_add=True, msk_add=True),
                T.ToTensor()
            ])
        else:
            print(f"Using deterministic transforms for {mode} mode.")
            self.transforms = T.Compose([
                T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
                T.RandomCrop(rand_crop=False, crop_size=SHAPE),
                T.ScaleIntensity(scope=SCOPE, range=RANGE),
                T.AddChannel(img_add=True, msk_add=True),
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Extract paths for the pair
        images = [sample["im0"], sample["im1"]]
        labels = [sample["msk0"], sample["msk1"]]

        # 2. Apply your original Transforms
        # This handles loading, cropping, scaling to [-1, 1], and PyTorch conversion
        images, labels = self.transforms(images, labels)

        i0 = images[0].float()
        i1 = images[1].float()
        m1 = labels[1].float()
        m0 = labels[0].float()

        # Compute blur (tenengrad) on raw intensity images before differencing
        # blur0 = tenengrad(i0.squeeze().numpy())
        # blur1 = tenengrad(i1.squeeze().numpy())
        blur0_x, blur0_y, blur0_z = tenengrad_per_axis(i0.squeeze().numpy())
        blur1_x, blur1_y, blur1_z = tenengrad_per_axis(i1.squeeze().numpy())


        # Rescale masks from [0, 1] to [-1, 1] to match image range
        m0 = m0 * 2.0 - 1.0
        m1 = m1 * 2.0 - 1.0

        # Save actual baseline image before calculating differences
        baseline_img = i0.clone()

        # 4. Apply Differences if flagged
        if self.diff:
            img1 = (i1 - i0) / 2.0  # Normalize difference to [-1, 1]

            # Sanity Check
            if not torch.allclose(img1 * 2.0 + i0, i1, atol=1e-6):
                print("⚠️ Difference img1 is incorrect!")

            # Overwrite i1 with the difference
            i1 = img1
            
        # 5. Get Delta T
        dt = torch.tensor([sample["dt"]], dtype=torch.float32)

        return {
            "baseline_img": baseline_img, # Securely the raw starting image of the pair
            "img0": i0,  # Shape: (1, D, H, W)
            "img1": i1,  # Raw target image OR the difference (if self.diff is True)
            "mask0": m0,  # Mask at time 0
            "mask1": m1,  # Mask at time 1
            "dt":  torch.tensor([sample["dt"]], dtype=torch.float32),
            "age0": torch.tensor([sample["age0"]], dtype=torch.float32),
            "gender": torch.tensor([sample["gender"]], dtype=torch.float32),
            "pair": torch.tensor([sample["pair"]], dtype=torch.int64),
            "blur0": torch.tensor([blur0_x, blur0_y, blur0_z], dtype=torch.float32),
            "blur1": torch.tensor([blur1_x, blur1_y, blur1_z], dtype=torch.float32),
            "PatientId": sample["PatientId"],
            "NoduleId": sample["NoduleId"],
            "t0": sample["t0"],
            "t1": sample["t1"],
        }


class NGPTrioDataset(Dataset):
    def __init__(self, root_dir, mode, dim, diff, num_folds=5, fold=0):
        """
        Args:
            root_dir (str): The root directory where images/json are stored.
            mode (str): 'train' or 'val' / 'test'.
            dim (int or tuple): The target spatial dimension (e.g., 64).
            diff (bool): Whether to output the differences or raw images.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.dim = dim
        self.diff = diff

        print(f"Diff is set to {self.diff} for NGPTrioDataset")

        # 1. Parse Data List using original logic
        if self.mode == 'train':
            data_list = get_data_list(root_dir, "training")
            tra_data, val_data = split_data_list(data_list, num_folds, fold=fold)
            self.data_list = tra_data
        elif self.mode == 'val':
            data_list = get_data_list(root_dir, "training")
            tra_data, val_data = split_data_list(data_list, num_folds, fold=fold)
            self.data_list = val_data
        else:
            self.data_list = get_data_list(root_dir, "test")

        # 2. Setup Preprocessing (Using your original custom Transforms!)
        SCOPE = (-1200, 600)
        RANGE = (-1.0, 1.0) # Explicitly forcing [-1, 1] range here
        SHAPE = [self.dim] * 3 if isinstance(self.dim, int) else self.dim

        if mode == 'train':
            self.transforms = T.Compose([
                T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
                T.RandomCrop(rand_crop=False, crop_size=SHAPE),
                T.ScaleIntensity(scope=SCOPE, range=RANGE),
                T.RandomFilp(prob=0.2, axes=(0, 1, 2)),
                T.RandomRot90(prob=0.2, axes=(0, 1, 2)),
                T.RandomScaleIntensity(prob=0.1, factor=0.1),
                T.RandomShiftIntensity(prob=0.1, offset=0.1*(RANGE[1]-RANGE[0])),
                T.AddChannel(img_add=True, msk_add=True),
                T.ToTensor()
            ])
        else:
            self.transforms = T.Compose([
                T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
                T.RandomCrop(rand_crop=False, crop_size=SHAPE),
                T.ScaleIntensity(scope=SCOPE, range=RANGE),
                T.AddChannel(img_add=True, msk_add=True),
                T.ToTensor()
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject = self.data_list[idx]
        info = subject["info"]
        
        # Time intervals
        months1 = float(info["Months1"]) # Interval T1 -> T2
        months2 = float(info["Months2"]) # Interval T2 -> T3

        age = float(info["Age"])     # Baseline age at T1
        gender = float(info["Gender"])  # Gender as float (0 for male, 1 for female)
        
        # 1. Extract paths (We pass the whole series to your original logic)
        images, labels = [], []
        for series in subject["series"]:
            images.append(os.path.join(self.root_dir, series["image"]))
            labels.append(os.path.join(self.root_dir, series["label"]))

        # 2. Apply your original Transforms
        # This handles loading, cropping, scaling to [-1, 1], and PyTorch conversion
        images, labels = self.transforms(images, labels)

        i0 = images[0].float()
        i1 = images[1].float()
        i2 = images[2].float()

        m0 = labels[0].float()
        m1 = labels[1].float()
        m2 = labels[2].float()

        # Compute blur (tenengrad) on raw intensity images before differencing
        blur0_x, blur0_y, blur0_z = tenengrad_per_axis(i0.squeeze().numpy())
        blur1_x, blur1_y, blur1_z = tenengrad_per_axis(i1.squeeze().numpy())
        blur2_x, blur2_y, blur2_z = tenengrad_per_axis(i2.squeeze().numpy())

        baseline_img = i0.clone()

        # Rescale masks from [0, 1] to [-1, 1] to match image range
        m0 = m0 * 2.0 - 1.0
        m1 = m1 * 2.0 - 1.0

        # 4. Apply Differences if flagged
        if self.diff:
            img0 = (i1 - i0) / 2.0  # Normalize difference to [-1, 1]
            img1 = (i2 - i0) / 2.0  
            img2 = (i2 - i1) / 2.0  

            # Overwrite i0, i1, i2 with the differences
            i0, i1, i2 = img0, img1, img2
            
        # 5. Get Delta T
        dt0 = torch.tensor([months1], dtype=torch.float32)  # T1 -> T2
        dt1 = torch.tensor([months2], dtype=torch.float32)  # T2 -> T3
        dt2 = torch.tensor([months1 + months2], dtype=torch.float32)  # T1 -> T3

        return {
            "baseline_img": baseline_img,
            "img0": i0,
            "img1": i1,
            "img2": i2,
            "dt0": dt0,
            "dt1": dt1,
            "dt2": dt2,
            "mask0": m0,
            "mask1": m1,
            "mask2": m2,
            "age0": torch.tensor([age], dtype=torch.float32),
            "gender": torch.tensor([gender], dtype=torch.float32),
            "PatientId": info["PatientId"],
            "NoduleId": info["NoduleId"],
            "t0": info["t0"],
            "t1": info["t1"],
            "blur0": torch.tensor([blur0_x, blur0_y, blur0_z], dtype=torch.float32),
            "blur1": torch.tensor([blur1_x, blur1_y, blur1_z], dtype=torch.float32),
            "blur2": torch.tensor([blur2_x, blur2_y, blur2_z], dtype=torch.float32),
        }


def get_data_list(data_root: str, key: str):
    json_path = os.path.join(data_root, "NGP3T.json")
    with open(json_path, "r") as jsf:
        json_data = json.load(jsf)[key]
    assert isinstance(json_data, list) and len(json_data) > 0

    csv_path = os.path.join(data_root, "patient-characteristics.csv")
    patient_dict = {}

    with open(csv_path, "r", newline='') as csvf:
        # 1. Use Sniffer to detect the delimiter (checks for , ; \t etc.)
        sample = csvf.read(2048)
        csvf.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;')
            delimiter = dialect.delimiter
        except csv.Error:
            # Fallback if sniffing fails
            delimiter = ',' 

        # 2. Use DictReader for cleaner mapping
        reader = csv.DictReader(csvf, delimiter=delimiter)
        
        for row in reader:
            # Assuming 'PatientId' is the first column header name
            # If the ID column name varies, you'd use row[reader.fieldnames[0]]
            subject_id = row.get("PatientId") or list(row.values())[0]
            patient_dict[subject_id] = row

    # 3. Match and transform
    filtered_data = []
    for item in json_data:
        subject_id = item["info"]["PatientId"]
        if subject_id in patient_dict:
            info = patient_dict[subject_id]
            
            # Map values safely
            item["info"]["Age"] = info.get("Age")
            
            gender = str(info.get("Gender", "")).lower()
            if gender == 'male':
                item["info"]["Gender"] = 0
            elif gender == 'female':
                item["info"]["Gender"] = 1
            
            filtered_data.append(item)
        else:
            print(f"WARNING: No data found for {subject_id} - removing from dataset.")

    return filtered_data



def split_data_list(data_list: list, num_folds: int, fold: int = 0):
    assert fold < num_folds, "`fold` should be less than `num_folds`."
    tra_data, val_data = [], []
    for idx, item in enumerate(data_list):
        if idx % num_folds == fold:
            val_data.append(item)
        else:
            tra_data.append(item)
    return tra_data, val_data