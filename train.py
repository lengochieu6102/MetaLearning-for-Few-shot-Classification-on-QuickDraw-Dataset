from argparse import ArgumentParser
import torch
import torchvision.transforms as transforms
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from quickdraw_dataset import QuickDraw
from model import QuicDrawCNN
from torch.utils.tensorboard import SummaryWriter
from utils import fast_adapt
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=10, help = "orgin model will be updated after number of batchsize sample task")
parser.add_argument("--num_epoch", type=int, default=50, help = "epoch")
parser.add_argument("--num_batch", type=int, default=100, help = "num_batch")
parser.add_argument("--ways", type=int, default=5, help = "number of class for each task")
parser.add_argument("--shots", type=int, default=5, help = "number of data for each class")
parser.add_argument("--numtasks", type=int, default=5000, help = "sample task in metadataset")
parser.add_argument("--fast_lr", type=float, default=0.005, help = "learning rate update in inner loop")
parser.add_argument("--lr", type=float, default=0.0005, help = "learning rate update in outer loop")
parser.add_argument("--adaptation_steps", type=int, default=5, help = "adaptation steps for each sample task")
parser.add_argument("--input_chanel", type=int, default=1, help = "input chanels of image")
parser.add_argument("--sample", type=int, default=20, help = "sample for each class in dataset of quickdraw (20 or 50 sample for experiment)")
parser.add_argument("--num_iteration", type=int, default=5000, help = "total number of iteration for learning to adapt few-shot")
parser.add_argument("--root", type=str, default='./', help = "root data folder of project")
args = parser.parse_args()


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.CenterCrop((128,128)),
    transforms.Resize((128,128))
     ])

train_ds = QuickDraw(root=args.root, transform= transform, mode='train',sample= args.sample)
valid_ds =  QuickDraw(root=args.root, transform= transform, mode='validation',sample= args.sample)
test_ds =  QuickDraw(root=args.root, transform= transform, mode='test',sample= args.sample)

meta_train = l2l.data.MetaDataset(train_ds)
meta_valid = l2l.data.MetaDataset(valid_ds)
meta_test = l2l.data.MetaDataset(test_ds)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mode_taskset(meta_dataset, ways, shots, numtasks):
    transforms_learn = [
        NWays(meta_dataset, n=ways),
        KShots(meta_dataset, k=shots*2),
        LoadData(meta_dataset),
        RemapLabels(meta_dataset),
        ConsecutiveLabels(meta_dataset)
    ]

    taskset = l2l.data.TaskDataset(dataset = meta_dataset, task_transforms = transforms_learn, num_tasks=numtasks)
    return taskset

train_taskset = mode_taskset(meta_train, args.ways, args.shots, args.numtasks)
valid_taskset = mode_taskset(meta_valid, args.ways, args.shots, args.numtasks)
test_taskset = mode_taskset(meta_test, args.ways, args.shots, args.numtasks)


features = QuicDrawCNN(args.input_chanel, args.shots, 64)
features = torch.nn.Sequential(features, l2l.nn.Lambda(lambda x: x.view(-1, 64*8*8)))
features.to(device)
head = torch.nn.Linear(64*8*8, args.ways)
head = l2l.algorithms.MAML(head, lr=args.fast_lr)
head.to(device)
all_parameters = list(features.parameters()) + list(head.parameters())
loss = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(all_parameters, lr=args.lr)

name_tb = f'{args.ways} way-{args.shots} shot-{args.sample} sample'
writer = SummaryWriter(f'runs/{name_tb}')

bar = tqdm(range(args.num_iteration))
for iteration in bar:
    optimizer.zero_grad()
    meta_train_error = 0
    meta_train_accuracy = 0
    meta_valid_error = 0 
    meta_valid_accuracy = 0
    for task in range(args.batch_size):
        # Compute meta-training loss
        learner = head.clone()
        batch = train_taskset.sample()
        query_error, query_accuracy = fast_adapt(batch, learner,features, loss, args.adaptation_steps, args.shots, args.ways, device)
        query_error.backward()
        meta_train_error += query_error.item()
        meta_train_accuracy += query_accuracy.item()
        # Average the accumulated gradients and optimize
        for p in all_parameters:
            p.grad.data.mul_(1.0 / args.batch_size)
        optimizer.step()

        # Compute meta-validation loss
        learner = head.clone()
        batch = test_taskset.sample()
        query_error, query_accuracy = fast_adapt(batch,learner,features, loss, args.adaptation_steps, args.shots, args.ways, device)
        meta_valid_error += query_error.item()
        meta_valid_accuracy += query_accuracy.item()

    train_loss = meta_train_error/args.batch_size
    train_acc = meta_train_accuracy/args.batch_size
    valid_loss = meta_valid_error/args.batch_size
    valid_acc = meta_valid_accuracy/args.batch_size
    bar.set_postfix({'train_loss': train_loss, 'train_acc': train_acc, 'valid_loss': valid_loss, 'valid_acc': valid_acc})
    writer.add_scalar('train_loss', train_loss, iteration)
    writer.add_scalar('train_acc', train_acc, iteration)
    writer.add_scalar('valid_loss', valid_loss, iteration)
    writer.add_scalar('valid_acc', valid_acc, iteration)
meta_test_error = 0.0
meta_test_accuracy = 0.0
for task in range(args.batch_size):
    # Compute meta-testing loss
    learner = head.clone()
    batch = test_taskset.sample()
    query_error, query_accuracy = fast_adapt(batch,learner,features, loss, args.adaptation_steps, args.shots, args.ways, device)
    meta_test_error += query_error.item()
    meta_test_accuracy += query_accuracy.item()
print('Meta Test Error', meta_test_error / args.batch_size)
print('Meta Test Accuracy', meta_test_accuracy / args.batch_size)
writer.add_hparams(
    hparam_dict= {'ways':args.ways, 'shots':args.shots, 'sample':args.sample}, 
    metric_dict = {'test_loss': meta_test_error / args.batch_size, 'test_acc':  meta_test_accuracy / args.batch_size}
)

    

    
    
    