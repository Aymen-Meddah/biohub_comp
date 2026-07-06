import torch 
import numpy as np 
def collate_fn(batch):
    images =[]
    targets = []
    datasets = []
    patches = []
    for sample in batch :
        image = sample["image"]
        if isinstance(image ,np.ndarray ):
            image = torch.from_numpy(image)
        image = image.float()
        image = image / image.max().clamo(min=1)

        image = image.unsqueeze(0)
        images.append(image)
        targets.append(sample["targets"])
        datasets.append(sample["dataset"])
        patches.append(sample["patch"])
    images =torch.stack(images)

    return {
        "image": images,

        "targets": targets,

        "dataset": datasets,

        "patch": patches

    }