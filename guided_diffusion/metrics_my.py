import torch
import torch.nn as nn

class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, model_output, xc_start ,mask):

        model_output=model_output.to("cuda:1")
        xc_start=xc_start.to("cuda:1")
        mask=mask.to("cuda:1")

        model_output=((model_output+1)/2).clamp(0, 1).to(torch.uint8)
        xc_start=((xc_start+1)/2).clamp(0, 1).to(torch.uint8)
        mask=((mask+1)/2).clamp(0, 1).to(torch.uint8)
        inputs=xc_start*mask
        outputs=model_output*mask
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())
        print("relevant:",relevant)
        print("selected:",selected)
        if relevant == 0 and selected == 0:
            return 1, 1

        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall


class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10
