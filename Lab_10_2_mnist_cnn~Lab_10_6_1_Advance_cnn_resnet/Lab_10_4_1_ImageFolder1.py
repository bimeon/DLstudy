import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader

from matplotlib.pyplot import imshow

trans = transforms.Compose([
    transforms.Resize((64,128))
])

train_data = torchvision.datasets.ImageFolder(root='origin_data', transform=trans)

for num, value in enumerate(train_data):
    data, label = value
    print(num, data, label)

    if (label == 0):
        data.save('train_data/gray/%d_%d.jpeg' % (num, label))
    else:
        data.save('train_data/red/%d_%d.jpeg' % (num, label))