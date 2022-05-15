import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from toy_datasets import *
import os
import argparse

HIDDEN_SIZES = [32, 32, 48, 48, 32]
parser = argparse.ArgumentParser()
parser.add_argument('--teacher_dir', default='./models/synthetic_teacher_tmp')
parser.add_argument('--temperature', default=1.)
parser.add_argument('--ntrain', default=15000)
parser.add_argument('--ntest', default=10000)
parser.add_argument('--n_epochs', default=125)
parser.add_argument('--lr', default=.001)
parser.add_argument('--eval_steps', default=1000)
parser.add_argument('--report_steps', default=50)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--dataset', default='sinusoid')
args = parser.parse_args()

teacher_model_dir = args.teacher_dir
TEMP = args.temperature
NTRAIN = args.ntrain
NTEST = args.ntest
NEPOCHS = args.n_epochs
LR = args.lr
REPORT_STEPS = args.report_steps
EVAL_STEPS = args.eval_steps
BATCH_SIZE = args.batch_size
dataset = args.dataset
# TODO, make this configurable somehow
class BaseMLP(nn.Module):
    def __init__(self, hidden_sizes, in_size, out_size):
        super(BaseMLP, self).__init__()
        layer_sizes = [in_size] + hidden_sizes + [out_size]
        layers = []

        for ii in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[ii], layer_sizes[ii+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(layer_sizes[ii+1]))
        layers = layers[:-1] # remove the activation before the final logits layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SyntheticDataset(Dataset):
    def __init__(self, xnumpy, y_hardlabels=None, yposterior=None, num_classes=2, estim_posterior=None, standardize=True,
                 mu=None, std=None): #pass mu=None for eval dataset
        self.x = xnumpy
        self.transform = transforms.ToTensor()
        if standardize:
            if mu is None:
                self.mu = np.mean(self.x, axis=0)
                self.std = np.std(self.x, axis=0)
            else:
                self.mu = mu
                self.std = std

            self.x_stdized = (self.x - self.mu)/self.std
            self.xdata = self.x_stdized
        else:
            self.xdata = self.x
        # if ylabel is None:
        self.targets = {}
        self.hardlabels = y_hardlabels.astype('int').reshape(-1, 1)
        self.one_hot_hardlabels = F.one_hot(torch.Tensor(self.hardlabels).to(torch.int64), num_classes=num_classes)
        self.posterior = None if yposterior is None else yposterior
        self.estim_posterior = None if estim_posterior is None else estim_posterior

    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, item):
        example = {}
        example['xdata'] = torch.Tensor(self.x_stdized[item])
        # print(self.hardlabels[item])
        # example['hardlabels'] = self.one_hot_hardlabels[item]
        example['hardlabels'] = torch.Tensor(self.hardlabels[item]).to(torch.int64)
        if self.posterior is not None:
            example['posterior'] =  torch.Tensor(self.posterior[item])
        if self.estim_posterior is not None:
            example['estim_posterior'] = torch.Tensor(self.estim_posterior[item])
        return example

def get_train_test_split():
    pass

def train_model_hard_labels(model, train_dataloader, test_dataloader, report_step, n_epochs, eval_step):
    print('********************************************************************')
    print('**************** START: Training Model w/ HardLabels ****************')
    model.train()
    xentropy_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    ''' create train dataloader'''
    # dataloader =
    ''' create test dataloder'''

    total_steps = len(train_dataloader) * n_epochs
    # with torch.no_grad():
    step = 0
    for epochnum in range(n_epochs):
        for ii, batchdata in enumerate(train_dataloader):
            xbatch = batchdata['xdata']
            ylabels = batchdata['hardlabels'].squeeze()
            output = model(xbatch)
            loss = xentropy_criterion(output, ylabels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ii % report_step==0:
                trainacc = accuracy(output, ylabels).item()
                lossval = loss.item()
                print('Step [%d/%d]| Epoch[%d/%d]| Train Acc [%.3f]| Train Loss [%.3f]' %
                      (step, total_steps, epochnum, n_epochs, trainacc, lossval))

            if ii % eval_step==0:
                print('--------------------- Running Evaluation ---------------------')
                results = run_evaluation(model=model, dataloader=test_dataloader)
                print('Step [%d/%d]| Epoch[%d/%d]| Test Acc [%.3f]| Test Loss [%.3f]' %
                      (step, total_steps, epochnum, n_epochs, results['accuracy'], results['label_xent']))
            step += 1

    results = run_evaluation(dataloader=test_dataloader, model=model)
    print('FINAL RESULTS')
    print('Test Acc [%.3f]: | Test Loss [%.3f]' % (results['accuracy'], results['label_xent']))
    print('**************** END: Training Model w/ SoftLabels ****************')
    print('********************************************************************')
    return model

def train_softlabels(model, train_dataloader, test_dataloader, batchlabel_key, n_epochs, eval_step, report_step, temperature):
    print('********************************************************************')
    print('**************** START: Training Model w/ SoftLabels ****************')
    print('Using %s as soft labels' % batchlabel_key)

    model.train()
    xentropy_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_dataloader) * n_epochs

    TARGETS_ARE_PROBAS = 'posterior' in batchlabel_key
    # with torch.no_grad():
    step = 0
    for epochnum in range(n_epochs):
        for ii, batchdata in enumerate(train_dataloader):
            xbatch = batchdata['xdata']
            yhardlabels = batchdata['hardlabels'].view(-1, 1)
            train_targets = batchdata[batchlabel_key]

            output = model(xbatch)

            with torch.no_grad():
                hard_label_xent = xentropy_criterion(output, yhardlabels)
                hard_label_xent = hard_label_xent.item()

            train_loss = loss_fn_smooth_labels(model_logits=output, target_values=train_targets,
                                               targetsprobas=TARGETS_ARE_PROBAS)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if ii % report_step==0:
                trainacc = accuracy(output, ylabels).item()
                lossval = train_loss.item()

                print('Step [%d/%d]| Epoch[%d/%d]| Train Loss (SoftLabel) [%.3f] |  Train Acc [%.3f]| Hard Label Xentropy [%.3f]' %
                      (step, total_steps, epochnum, n_epochs, lossval, trainacc, hard_label_xent))

            if ii % eval_step==0:
                print('--------------------- Running Evaluation ---------------------')
                results = run_evaluation(model=model, dataloader=test_dataloader)
                print('Step [%d/%d]| Epoch[%d/%d]| Test Acc [%.3f]| Test Loss [%.3f]' %
                      (step, total_steps, epochnum, n_epochs, results['accuracy'], results['label_xent']))
            step += 1

    results = run_evaluation(dataloader=test_dataloader, model=model)
    print('FINAL RESULTS')
    print('Test Acc [%.3f]: | Test Loss [%.3f]' % (results['accuracy'], results['label_xent']))
    print('**************** END: Training Model w/ SoftLabels ****************')
    print('********************************************************************')

def get_model_logits(model, dataloader):
    model.eval()
    model_logits = []
    model_probas = []
    with torch.no_grad():
        for ii, (batchdata) in enumerate(dataloader):
            xbatch = batchdata['xdata']
            output = model(xbatch)
            model_logits.append(output.detach().cpu().numpy())
            model_probas.append(F.softmax(output, dim=1).detach().cpu().numpy())
    model.train()

    return np.concatenate(model_logits), np.concatenate(model_probas)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))

    return res[0]

def estimate_posterior(xdata, hard_labels):
    '''
    Use KNN posterior estimation algorithm to get posterior labels
    :xdata: this can be either the data itself, or for larger datasets the autoencoded embeddings
    :hard_labels: hard labels can be from the teacher model or from the ground truth hard labels
    '''

    pass


def run_evaluation(dataloader, model):
    results = {}
    xentropy = nn.CrossEntropyLoss()
    ndata = 0
    ncorrect=0
    example = dataloader.dataset[0]
    datakeys = list(example.keys())
    results['accuracy'] = []
    results['label_xent'] = []
    if 'estim_posterior' in datakeys:
        results['posterior_xent'] = []

    if 'posterior' in datakeys:
        results['estim_posterior_xent'] = []

    for ii, batchdata in enumerate(dataloader):
        xbatch = batchdata['xdata']
        nbatch = xbatch.size()[0]

        model_output = model(xbatch)
        labels = batchdata['hardlabels'].squeeze()
        hard_xent = xentropy(model_output, labels)
        results['label_xent'].append(hard_xent * nbatch)
        _ncorrect =  accuracy(model_output, target=labels).item() * nbatch
        ncorrect += _ncorrect
        ndata += nbatch

        # results['accuracy'].append(ncorrect)
        #TODO: evaluate the following
        # 1. Hard Xent
        # 2. Accuracy
        # 3. Soft label xent
        # 4. Estimated posterior xent
        if 'posterior' in datakeys:
            yposterior = batchdata['posterior']
            kldiv = loss_fn_kd(student_logits=model_output, teacher_logits=yposterior, T=1.)
            kd_loss = loss_fn_kd(student_logits=model_output, teacher_logits=yposterior, T=TEMP)
            # posterior_xent = loss_fn_kd(model_output, yposterior)

            results['posterior_kldiv'] = kldiv
            results['posterior_kd_loss'] =  kd_loss

        if 'estim_posterior' in datakeys:
            estim_posterior = batchdata['posterior']
            kldiv = loss_fn_kd(student_logits=model_output, teacher_logits=estim_posterior, T=1.)
            kd_loss = loss_fn_kd(student_logits=model_output, teacher_logits=estim_posterior, T=TEMP)
            results['estim_posterior_kldiv'] = kldiv
            results['estim_posterior_kd_loss'] = kd_loss

    # results['accuracy'] = sum(results['accuracy']) / ndata
    results['accuracy'] = ncorrect / ndata
    results['label_xent'] = sum(results['label_xent']) / ndata

    return results

def train_pytorch_model(model, xdata, ylabels, optimizer,
                        train_iters=1000, learning_rate=.001, temperature=1):
    kd_loss = loss_fn_kd

    # create train dataloader

    # create test dataloader

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    n_iters = 0
    while n_iters < train_iters:
        for ii, (xbatch, y_estimated_probas) in range(train_iters): # y_estimated_probas should be
            model_outputs = model(xbatch)
            loss = kd_loss(student_logits=model_outputs, teacher_logits= y_estimated_probas, T=temperature)
            optimizer.zero_grad()
            loss.backward()
            n_iters += 1
            if n_iters > train_iters:
                break
            n_iters += 1
        #shuffle dataloader
    pass

def train_active(model, xdata, ylabels, train_iters, learning_rate, temperature):


    # step 1: generate dataset

    # step 2:

    # step 3:

    # step 4:

    # step 5:

    pass

def calculate_posterior(model, xdata):
    pass

def calculate_loss(model, xdata, ylabels):
    pass

#TODO: put the following into it's own script-----------------------------------------
# call it losses.py or something

#KL Divergence loss - grab this function from the quantization code
def loss_fn_kd(student_logits, teacher_logits, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs from student and teacher
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2

    student_inputs can either be 'logits' or 'probas'
    teacher_inputs can either be 'logits' or 'probas'
    """


    teacher_soft_logits = F.softmax(teacher_logits / T, dim=1)
    teacher_soft_logits = teacher_soft_logits.float()

    student_soft_logits = F.log_softmax(student_logits/T, dim=1)


    #For KL(p||q), p is the teacher distribution (the target distribution), and
    KD_loss = nn.KLDivLoss(reduction='batchmean')(student_soft_logits, teacher_soft_logits)
    KD_loss = (T ** 2) * KD_loss
    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)

    return KD_loss


def loss_fn_smooth_labels(model_logits, target_values, targetsprobas=False):
    if not targetsprobas:
        target_probas = torch.softmax(target_values, dim=1)
    else:
        target_probas = target_values

    model_probs = F.log_softmax(model_logits, dim=1)
    smooth_label_loss = nn.KLDivLoss(reduction='batchmean')(model_probs, target_probas)
    return smooth_label_loss


if __name__ == "__main__":
    teacher_model_filepath = os.path.join(teacher_model_dir, 'teacher.mdl')

    # Load Data
    # from classifier_utils import *
    from densityestimate import *
    # print('asdf')
    def generate_samples(n_samples):
        xdata = 10 * (np.random.rand(n_samples, 2)) - 4
        y_posterior = np.array([oracle_sinusoid_posterior(xdata[_ii]) for _ii in range(n_samples)])
        ylabel = np.array(y_posterior>.5).astype('float32')
    #     ylabel = np.hstack([ylabel, np.abs(1-ylabel)])
        return xdata, y_posterior, ylabel

    ''' Step 1: Generate training/testing data'''
    if dataset=='gaussian':
        xdata0, xdata1, ylabels = gen_simple_binary_data(mu_x=0, mu_y=4, N=10000, ndims=2)
        xtrain = np.vstack([xdata0, xdata1])

        xdata0, xdata1, ytest = gen_simple_binary_data(mu_x=0, mu_y=4, N=1000, ndims=2)
        xtest = np.vstack([xdata0, xdata1])

    elif dataset=='sinusoid':
        xtrain, y_posterior, ylabels = generate_samples(NTRAIN)
        xtest, y_posterior_test, ytest = generate_samples(NTEST)

        y_posterior = np.vstack([1-y_posterior, y_posterior]).T
        y_posterior_test = np.vstack([1-y_posterior_test, y_posterior_test]).T

    ''' Step 2: Estimate posterior '''

    full_train_dataset = SyntheticDataset(xnumpy=xtrain, y_hardlabels=ylabels, yposterior=None,
                                             estim_posterior=None, standardize=True, mu=None, std=None)
    # full_train_dataset.shuffle()
    test_dataset = SyntheticDataset(xnumpy=xtest, y_hardlabels=ytest, yposterior=None,
                                          estim_posterior=None, standardize=True,
                                          mu=full_train_dataset.mu, std=full_train_dataset.std)
    # test_dataset.shuffle()
    train_dataloader = torch.utils.data.DataLoader(dataset=full_train_dataset, batch_size=BATCH_SIZE,
                                                   num_workers=4, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                                 num_workers=4, pin_memory=True)

    ''' Step 3: Train Teacher model if not exists'''
    teacher_model = BaseMLP(hidden_sizes=HIDDEN_SIZES, in_size=2, out_size=2)
    if not os.path.isdir(teacher_model_dir):
        os.makedirs(teacher_model_dir, exist_ok=True)

    if not os.path.exists(teacher_model_filepath):
        trained_teacher_model = train_model_hard_labels(model=teacher_model, train_dataloader=train_dataloader,
                                                        test_dataloader=test_dataloader, report_step=REPORT_STEPS,
                                                        eval_step=EVAL_STEPS, n_epochs=NEPOCHS)
        # trained_teacher_model.save(teacher_model_filepath)
        torch.save({'model_state_dict': trained_teacher_model.state_dict()}, teacher_model_filepath)
    else:
        checkpoint = torch.load(teacher_model_filepath)
        trained_teacher_model = BaseMLP(hidden_sizes=HIDDEN_SIZES, in_size=2, out_size=2)
        trained_teacher_model.load_state_dict(checkpoint['model_state_dict'])

    '''Step 4: Train student with full dataset'''
    student_model = BaseMLP(hidden_sizes=HIDDEN_SIZES, in_size=2, out_size=2)

    trlogits, trprobas = get_model_logits(dataloader=test_dataloader, model=trained_teacher_model)
    telogits, teprobas = get_model_logits(dataloader=test_dataloader, model=trained_teacher_model)
    soft_train_dataset = SyntheticDataset(xnumpy=xtrain, y_hardlabels=ylabels, yposterior=trprobas, estim_posterior=None,
                                          standardize=True, mu=None, std=None)
    soft_test_dataset = SyntheticDataset(xnumpy=xtest, y_hardlabels=ylabels, yposterior=teprobas, estim_posterior=None,
                                         standardize=True, mu=soft_train_dataset.mu, std=soft_train_dataset.std)
    soft_train_dataloader = torch.utils.data.DataLoader(dataset=soft_train_dataset, batch_size=BATCH_SIZE, num_workers=4,
                                                        pin_memory=True)
    soft_test_dataloader = torch.utils.data.DataLoader(dataset=soft_test_dataset, batch_size=BATCH_SIZE, num_workers=4,
                                                       pin_memory=True)
    trained_student_model = train_softlabels(model=student_model, train_dataloader=soft_train_dataloader,
                                             test_dataloader=soft_test_dataloader, batchlabel_key='posterior',
                                             n_epochs=NEPOCHS, eval_step=EVAL_STEPS, report_step=REPORT_STEPS,
                                             temperature=1)
    '''Step 5: Active train student model'''


    pass




