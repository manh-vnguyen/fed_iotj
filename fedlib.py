#%%
import torch

class Client():
    def __init__(self, model_type, model, trainloader, loss_fn, device, checkpointed_epoch=0):
        self.model_type = model_type
        self.device = device
        self.model = model
        self.trainloader = trainloader
        self.loss_fn = loss_fn
        self.data_iter = iter(self.trainloader)
        for _ in range(checkpointed_epoch % len(self.trainloader)):
            next(self.data_iter)

    def train(self):
        try:
            x, y = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.trainloader)
            x, y = next(self.data_iter)

        x, y = x.to(self.device), y.to(self.device)
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.loss = loss.item()
        self.num_samples = y.shape[0]

def test_model(model, loader, device):
    acc, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acc += torch.sum(torch.argmax(logits, dim=1) == y).item()
            total += y.shape[0]
    test_acc = acc/total*100
    return test_acc

def eval_model(model, loader, device, criterion=torch.nn.CrossEntropyLoss()):
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def extract_grads(model):
    return torch.cat([p.grad.view(-1) for p in model.parameters()])

def apply_grads(update, model):
    offset = 0
    for p in model.parameters():
        p.grad = update[offset : offset + p.numel()].view_as(p).detach().clone()
        offset += p.numel()

def extract_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def apply_params(flat_params, model):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel

def extract_bn_stats(modules):
    return torch.cat([t.reshape(-1) for m in modules for t in (m.running_mean, m.running_var)])

def apply_bn_stats(flat_stats, modules):
    offset = 0
    for m in modules:
        n = m.running_mean.numel()

        m.running_mean.copy_(flat_stats[offset:offset + n].view_as(m.running_mean))
        offset += n

        m.running_var.copy_(flat_stats[offset:offset + n].view_as(m.running_var))
        offset += n
