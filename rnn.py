import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score


SOS, EOS, PAD = '^', '$', ' '
PAD_ID = 0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_vocab(names):
    # пробел должен иметь id 0, чтобы его можно было игнорировать в лоссе
    letters = sorted(set(''.join(names)))
    tokens = [PAD, SOS, EOS] + letters
    int_to_token = {i: t for i, t in enumerate(tokens)}
    token_to_int = {t: i for i, t in enumerate(tokens)}
    return int_to_token, token_to_int


def encode_names(names, token_to_int, max_len):
    # ^name$ + пробелы до max_len
    seqs = []
    for name in names:
        s = SOS + name + EOS
        s = s + PAD * (max_len - len(s))
        seqs.append([token_to_int[ch] for ch in s])
    return np.array(seqs)


class customDataset(Dataset):
    def __init__(self, X, y):
        # X: массив (n, max_len), цель для буквы t это буква t+1
        self.X = X
        self.X_next = np.concatenate([X[:, 1:], np.full((len(X), 1), PAD_ID)], axis=1)
        self.y = np.array(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.X_next[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


# ---------- рекуррентные ячейки ----------

class custom_VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_h = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden):
        return torch.tanh(self.W_h(x) + self.W_hh(hidden))


class custom_GRU_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), dim=1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        h_tilda = torch.tanh(self.W(torch.cat((x, r * hidden), dim=1)))
        return (1 - z) * hidden + z * h_tilda


class custom_LSTM_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden):
        h, c = hidden
        combined = torch.cat((x, h), dim=1)
        f = torch.sigmoid(self.W_f(combined))
        i = torch.sigmoid(self.W_i(combined))
        c_tilde = torch.tanh(self.W_c(combined))
        c = f * c + i * c_tilde
        o = torch.sigmoid(self.W_o(combined))
        h = o * torch.tanh(c)
        return h, c


# ---------- модель ----------

class custom_RNN(nn.Module):
    def __init__(self, size_letter=16, size_hidden=32, rnn_block=custom_VanillaRNN,
                 n_tokens=29, n_outputs=1, p=0.3, p1=0.3):
        super().__init__()
        self.size_hidden = size_hidden
        self.embedding = nn.Embedding(n_tokens, size_letter)
        self.rnn_cell = rnn_block(size_letter, size_hidden)
        self.dropout = nn.Dropout(p)
        self.next_token = nn.Linear(size_hidden, n_tokens)
        self.gender_classification = nn.Sequential(
            nn.Linear(size_hidden, size_hidden // 2),
            nn.ReLU(),
            nn.Dropout(p1),
            nn.Linear(size_hidden // 2, n_outputs)
        )

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(batch_size, self.size_hidden, device=device)
        if isinstance(self.rnn_cell, custom_LSTM_RNN):
            return hidden, torch.zeros(batch_size, self.size_hidden, device=device)
        return hidden

    def step(self, x_t, hidden):
        # один шаг ячейки, возвращает новое состояние и h (для LSTM это первая часть пары)
        hidden = self.rnn_cell(x_t, hidden)
        h = hidden[0] if isinstance(hidden, tuple) else hidden
        return hidden, h

    def forward(self, x):
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        hidden = self.init_hidden(batch_size, x.device)
        token_logits = []
        for t in range(seq_len):
            hidden, h = self.step(embedded[:, t, :], hidden)
            token_logits.append(self.next_token(h))
        token_logits = torch.stack(token_logits, dim=1)
        # пол предсказываем по последнему состоянию
        gender_logits = self.gender_classification(self.dropout(h))
        return token_logits, gender_logits

    def get_name_embedding(self, x):
        # эмбеддинг имени = последнее скрытое состояние
        embedded = self.embedding(x)
        hidden = self.init_hidden(x.shape[0], x.device)
        for t in range(x.shape[1]):
            hidden, h = self.step(embedded[:, t, :], hidden)
        return h


class TorchRNN(nn.Module):
    # та же модель, но на готовых слоях nn.RNN / nn.GRU / nn.LSTM (для бонуса)
    def __init__(self, size_letter=16, size_hidden=32, cell='gru', n_tokens=29,
                 bidirectional=False, num_layers=1, p=0.3):
        super().__init__()
        layers = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}
        self.embedding = nn.Embedding(n_tokens, size_letter)
        self.rnn = layers[cell](size_letter, size_hidden, batch_first=True, num_layers=num_layers,
                                bidirectional=bidirectional, dropout=p if num_layers > 1 else 0)
        self.bidirectional = bidirectional
        out_size = size_hidden * 2 if bidirectional else size_hidden
        self.dropout = nn.Dropout(p)
        # двунаправленная сеть видит следующую букву, поэтому голову для букв делаем только для обычной
        self.next_token = None if bidirectional else nn.Linear(out_size, n_tokens)
        self.gender_classification = nn.Sequential(
            nn.Linear(out_size, out_size // 2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(out_size // 2, 1)
        )

    def get_name_embedding(self, x):
        rnn_out, h_n = self.rnn(self.embedding(x))
        if isinstance(h_n, tuple):  # у LSTM (h, c)
            h_n = h_n[0]
        if self.bidirectional:
            return torch.cat((h_n[-2], h_n[-1]), dim=1)
        return h_n[-1]

    def forward(self, x):
        rnn_out, _ = self.rnn(self.embedding(x))
        token_logits = self.next_token(rnn_out) if self.next_token is not None else None
        gender_logits = self.gender_classification(self.dropout(self.get_name_embedding(x)))
        return token_logits, gender_logits


# ---------- обучение ----------

def token_loss(loss_fn, token_logits, y_next):
    # последнюю позицию не берём: после неё букв нет
    n_tokens = token_logits.shape[-1]
    return loss_fn(token_logits[:, :-1].reshape(-1, n_tokens), y_next[:, :-1].reshape(-1))


def gender_loss(loss_fn, gender_logits, y, loss_name='BCE'):
    if loss_name == 'BCE':
        return loss_fn(gender_logits.squeeze(1), y)
    # NLL: два выхода, log_softmax по классам
    return loss_fn(F.log_softmax(gender_logits, dim=1), y.long())


def train_loop(dataloader, model, loss_fn, optimizer, device, pred='next_token', loss_name='BCE'):
    model.train()
    train_loss = 0
    for X, y_next, y in dataloader:
        X, y_next, y = X.to(device), y_next.to(device), y.to(device)
        token_logits, gender_logits = model(X)
        if pred == 'next_token':
            loss = token_loss(loss_fn, token_logits, y_next)
        else:
            loss = gender_loss(loss_fn, gender_logits, y, loss_name)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)


def evaluate(dataloader, model, loss_fn, device, pred='next_token', loss_name='BCE'):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for X, y_next, y in dataloader:
            X, y_next, y = X.to(device), y_next.to(device), y.to(device)
            token_logits, gender_logits = model(X)
            if pred == 'next_token':
                loss = token_loss(loss_fn, token_logits, y_next)
            else:
                loss = gender_loss(loss_fn, gender_logits, y, loss_name)
            valid_loss += loss.item()
    return valid_loss / len(dataloader)


def train_loop_both(dataloader, model, loss_fn_token, loss_fn_gender, optimizer, device, lam1=1.0, lam2=1.0):
    model.train()
    train_loss = 0
    for X, y_next, y in dataloader:
        X, y_next, y = X.to(device), y_next.to(device), y.to(device)
        token_logits, gender_logits = model(X)
        loss = lam1 * token_loss(loss_fn_token, token_logits, y_next) + \
               lam2 * gender_loss(loss_fn_gender, gender_logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)


def evaluate_both(dataloader, model, loss_fn_token, loss_fn_gender, device):
    model.eval()
    loss_token, loss_gender = 0, 0
    with torch.no_grad():
        for X, y_next, y in dataloader:
            X, y_next, y = X.to(device), y_next.to(device), y.to(device)
            token_logits, gender_logits = model(X)
            loss_token += token_loss(loss_fn_token, token_logits, y_next).item()
            loss_gender += gender_loss(loss_fn_gender, gender_logits, y).item()
    return loss_token / len(dataloader), loss_gender / len(dataloader)


def predict_gender(dataloader, model, device, loss_name='BCE'):
    # вероятность того, что имя мужское
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y_next, y in dataloader:
            _, gender_logits = model(X.to(device))
            if loss_name == 'BCE':
                probs = torch.sigmoid(gender_logits).squeeze(1)
            else:
                probs = F.softmax(gender_logits, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(y)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def evaluate_class(dataloader, model, device, loss_name='BCE'):
    probs, labels = predict_gender(dataloader, model, device, loss_name)
    return roc_auc_score(labels, probs)


# ---------- генерация ----------

def generate_name(model, int_to_token, token_to_int, start_seq='', temperature=1.0, max_len=13, device='cpu'):
    model.eval()
    start_seq = SOS + start_seq.lower()
    with torch.no_grad():
        hidden = model.init_hidden(1, device)
        int_seq = torch.tensor([token_to_int[ch] for ch in start_seq], device=device).unsqueeze(0)
        embedded = model.embedding(int_seq)
        # прогоняем начало имени
        for t in range(len(start_seq)):
            hidden, h = model.step(embedded[:, t, :], hidden)

        result = start_seq[1:]
        for _ in range(max_len - len(start_seq)):
            logits = model.next_token(h)
            probs = F.softmax(logits / temperature, dim=-1).squeeze(0)
            next_idx = torch.multinomial(probs, num_samples=1)
            next_char = int_to_token[next_idx.item()]
            if next_char in [EOS, PAD]:
                break
            result += next_char
            hidden, h = model.step(model.embedding(next_idx), hidden)
    return result


# ---------- перплексия ----------

def perplexity_from_probs(dataloader, model, device):
    # exp(среднее -log p(правильная буква)), пробелы после конца имени не считаем
    model.eval()
    log_probs = []
    with torch.no_grad():
        for X, y_next, y in dataloader:
            token_logits, _ = model(X.to(device))
            probs = torch.softmax(token_logits[:, :-1], dim=-1).cpu()
            target = y_next[:, :-1]
            p = probs.gather(2, target.unsqueeze(2)).squeeze(2)
            log_probs.append(torch.log(p[target != PAD_ID]))
    return torch.exp(-torch.cat(log_probs).mean()).item()


def perplexity_from_loss(loss):
    # средний cross entropy по батчам -> PPL (чуть отличается, потому что батчи разной длины)
    return np.exp(loss)


class FreqModel():
    # "модель популярных букв": всегда возвращает частоты букв из train
    def __init__(self, n_tokens=29):
        self.n_tokens = n_tokens
        self.freq_probs = None

    def train(self, X_next):
        targets = X_next[:, :-1].reshape(-1)
        targets = targets[targets != PAD_ID]
        counts = np.bincount(targets, minlength=self.n_tokens)
        self.freq_probs = counts / counts.sum()


def perplexity_frequency(model, dataloader):
    log_probs = []
    for X, y_next, y in dataloader:
        target = y_next[:, :-1].reshape(-1)
        target = target[target != PAD_ID].numpy()
        log_probs.append(np.log(model.freq_probs[target]))
    return np.exp(-np.concatenate(log_probs).mean())


# ---------- затухание градиентов ----------

def grad_by_position(model, dataloader, loss_fn, optimizer, device, pred='gender', epochs=5):
    # учим модель и после каждого батча смотрим градиент лосса по эмбеддингу каждой позиции
    # результат: (epochs, max_len) - средняя норма градиента по позиции
    model = model.to(device)
    grads = []

    def backward_hook(module, grad_input, grad_output):
        # grad_output[0]: (batch, max_len, size_letter)
        grads.append(grad_output[0].detach().norm(dim=-1).mean(dim=0).cpu())

    handle = model.embedding.register_full_backward_hook(backward_hook)
    epoch_grads = []
    for epoch in range(epochs):
        grads.clear()
        train_loop(dataloader, model, loss_fn, optimizer, device, pred=pred)
        epoch_grads.append(torch.stack(grads).mean(dim=0))
    handle.remove()
    return torch.stack(epoch_grads)


def fit(model, train_dl, valid_dl, loss_fn, device, pred='next_token', epochs=30, lr=1e-3, loss_name='BCE'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        train_losses.append(train_loop(train_dl, model, loss_fn, optimizer, device, pred, loss_name))
        valid_losses.append(evaluate(valid_dl, model, loss_fn, device, pred, loss_name))
    return train_losses, valid_losses


def fit_both(model, train_dl, valid_dl, loss_fn_token, loss_fn_gender, device, epochs=25, lr=1e-3, lam1=1.0, lam2=1.0):
    # multitask: возвращает лоссы по буквам и по полу отдельно для train и valid
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_token': [], 'train_gender': [], 'valid_token': [], 'valid_gender': []}
    for epoch in range(epochs):
        train_loop_both(train_dl, model, loss_fn_token, loss_fn_gender, optimizer, device, lam1, lam2)
        tr_token, tr_gender = evaluate_both(train_dl, model, loss_fn_token, loss_fn_gender, device)
        vl_token, vl_gender = evaluate_both(valid_dl, model, loss_fn_token, loss_fn_gender, device)
        history['train_token'].append(tr_token)
        history['train_gender'].append(tr_gender)
        history['valid_token'].append(vl_token)
        history['valid_gender'].append(vl_gender)
    return history
