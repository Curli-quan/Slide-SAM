from tutils.nn.data import write, np_to_itk
import numpy as np


class Masks3D:
    def __init__(self) -> None:
        pass

    def from_dict(self, masks):
        self.masks = masks
        # self.tags = masks

    def to_2dmasks(self):
        pass

    def filter_by_bg(self, volume, threshold=None):
        threshold = (volume.max() - volume.min()) * 0.1 + volume.min()
        keys = self.masks.keys()
        for k in keys:
            v = self.masks[k]
            assert v.shape == volume.shape, f"Got shape ERROR, {v.shape, volume.shape}"
            if (v * volume).mean() <= threshold:
                self.masks.pop(k)
    
    # def filter_by_area(self,):


    def sort_by_logits(self):
        self.confidences = []
        self.tags_by_conf = []
        for k, v in self.masks.items():
            confidence = v[v>0].mean()
            self.confidences.append(confidence)
            self.tags_by_conf.append(k)
        indices = np.argsort(self.confidences)[::-1]
        self.tags_by_conf = np.array(self.tags_by_conf)[indices].tolist()
        self.confidences = np.array(self.confidences)[indices]

    def to_nii(self, path="tmp.nii.gz"):
        self.sort_by_logits()
        total = None
        for i, k in enumerate(self.tags_by_conf):
            mask = np.int32(self.masks[k]>0)
            if total is None:
                total = mask * i
            else:
                total = np.where(total>0, total, mask * i)
        mask_itk = np_to_itk(total)
        write(mask_itk, path)


if __name__ == "__main__":
    p = "/home1/quanquan/code/projects/finetune_large/segment_anything/tmp.npy"
    data = np.load(p, allow_pickle=True).tolist()
    # import ipdb; ipdb.set_trace()
    print(data.keys())
    mm = Masks3D()
    mm.from_dict(data)
    mm.to_nii()