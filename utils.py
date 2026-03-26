#%%
import torch
from torch import nn
import json



class BatchNorm(nn.Module):
    VARIANCE_UNBIASED = True # False will make it identical to nn.LayerNorm
    def __init__(self, num_channels, momentum=0.1, eps=1e-5):
        super().__init__()
        self.repr = f"BatchNorm({num_channels=}, {momentum=}, {eps=})"
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels).float())
        self.beta = nn.Parameter(torch.zeros(num_channels).float())
        self.register_buffer('running_mean', torch.zeros(num_channels)) 
        self.register_buffer('running_var', torch.ones(num_channels))

    def __repr__(self):
        return self.repr
    
    def forward(self, x): # (B, C, *)
        if self.training:
            mean = x.transpose(0, 1).mean(dim=list(range(1, x.ndim)), keepdim=True).transpose(0, 1) # (1, C, 1, 1)
            var = x.transpose(0, 1).var(dim=list(range(1, x.ndim)), keepdim=True, unbiased=self.VARIANCE_UNBIASED).transpose(0, 1) # (1, C, 1, 1)
            x = (x - mean) / torch.sqrt(var + self.eps) # transform the mean and var into (1, C, 1, 1) for broadcasting semanticj
            x = x * self.gamma.view([-1]+[1] * (x.ndim - 2)) + self.beta.view([-1]+[1] * (x.ndim - 2))
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.view(-1))
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.view(-1))
        else:
            x = (x - self.running_mean.view([-1]+[1] * (x.ndim - 2))) / torch.sqrt(self.running_var.view([-1]+[1] * (x.ndim - 2)) + self.eps) # transform the mean and var into (1, C, 1, 1) for broadcasting semantic
            x = x * self.gamma.view([-1]+[1] * (x.ndim - 2)) + self.beta.view([-1]+[1] * (x.ndim - 2))
        return x

class BNTT(nn.Module):
    def __init__(self, T, num_channels, **kwargs):
        super().__init__()
        self.bntt = nn.ModuleList([BatchNorm(num_channels, **kwargs) for _ in range(T)])
        self.t = 0

    def reset(self):
        self.t = 0

    def forward(self, x):
        out = self.bntt[self.t](x)
        self.t += 1
        return out

def get_best_gpu(min_required):
    import os
    max_arg = lambda a: (max(a), a.index(max(a)))
    free, gpu = max_arg(list(map(int, os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits').read().strip().split('\n'))))
    if free >= min_required:
        return gpu
    raise Exception("Not enough resources")

class ARGS():
    def __init__(self, inp):
        for key in inp.keys():
            setattr(self, key, inp[key])
        self.repr = inp
    
    def __repr__(self):
        return str(self.repr)


from mysql.connector import pooling
from datetime import datetime, timezone
import os
import dotenv

dotenv.load_dotenv()

MYSQL_CONFIG = dict(
    host=f"{os.getenv('SQL_HOST')}",
    port=3306,
    user=f"{os.getenv('SQL_USER')}",
    password=f"{os.getenv('SQL_PASS')}",
    database="iotj",
    autocommit=False,
)

_POOL = None

def _get_pool():
    global _POOL
    if _POOL is None:
        # make it small because each process gets its own pool
        _POOL = pooling.MySQLConnectionPool(
            pool_name=f"iotj_{os.getpid()}",
            pool_size=1,          # or 2 max, usually enough for a logging writer
            **MYSQL_CONFIG
        )
    return _POOL

def _get_con():
    return _get_pool().get_connection()

import re

_ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _ident(name: str) -> str:
    # Prevent SQL injection via table/column names
    if not _ident_re.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return f"`{name}`"  # MySQL identifier quoting

def write_run_to_db(table: str, data: dict):
    if not data:
        raise ValueError("data must not be empty")

    table_sql = _ident(table)
    columns = list(data.keys())

    cols_sql = ", ".join(_ident(c) for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    values = tuple(data[c] for c in columns)

    query = f"INSERT INTO {table_sql} ({cols_sql}) VALUES ({placeholders})"

    con = _get_con()
    try:
        with con.cursor() as cur:
            cur.execute(query, values)
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

class RunningAverage:
    """O(1) update running average."""
    __slots__ = ("_count", "_total")

    def __init__(self):
        self._count = 0
        self._total = 0.0

    def add(self, value: float) -> float:
        self._count += 1
        self._total += float(value)
        return self.value

    @property
    def count(self) -> int:
        return self._count

    @property
    def total(self) -> float:
        return self._total

    @property
    def value(self) -> float:
        return self._total / self._count if self._count else 0.0

    def reset(self) -> None:
        self._count = 0
        self._total = 0.0

import time

class TrackSampling:
    """O(1) update running average."""
    __slots__ = ("_count_samples", "_start_time")

    def __init__(self):
        self.reset()

    def reset(self):
        self._count_samples = 0
        self._start_time = time.time()

    def add(self, value: float) -> float:
        self._count_samples += value

    @property
    def sps(self) -> float:
        return self._count_samples / (time.time() - self._start_time)

    @property
    def count(self) -> int:
        return self._count_samples