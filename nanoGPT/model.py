import torch
import torch.nn as nn
from config import GPTConfig as Config
from torch.nn import functional as F
import inspect


# Root Mean Square Layer Normalization https://arxiv.org/abs/1910.07467
# 使用平方根均方归一化，降低噪声的影响
# 可以认为是LayerNorm的一种改进
# 参考LLaMA 3 https://github.com/meta-llama/llama3/blob/main/llama/model.py
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# flash attention
class FlashAttention(nn.Module):
    def __init__(
            self,
            c: Config
    ):
        super().__init__()
        self.n_head = c.n_head
        self.n_embed = c.n_embed
        self.head_size = c.n_embed // c.n_head
        self.dropout = c.dropout
        # 这里将W_Q,W_K,W_V合并到一个Linear里面去,及就是concat(Q,K,V) = W[X]
        self.qkv = nn.Linear(c.n_embed, 3 * c.n_embed, bias=c.bias)
        self.att_dropout = nn.Dropout(c.dropout)
        self.att_proj = nn.Linear(c.n_embed, c.n_embed, bias=c.bias)

    def forward(self, x):
        # Batch, Sequence, Embedding Dimensionality
        B, T, C = x.shape
        # Split Q, K, V, now shape is B, T, C
        q, k, v = self.qkv(x).split(self.n_embed, dim=2)
        # Reshape Q, K, V.   C = n_head * head_size = bh * hs
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # B, nh, T, hs
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Use flash attention
        # 注意这里的代码是封装好的, pytorch >= 2.0.0版本封装了flash attention的优化代码
        # Q(B, nh, T, hs) x K(B, nh, hs, T) -> ATT(B, nh, T, hs)
        # ATT ---softmax---> ATT(B, nh, T, T) x V(B, nh, T, hs) -> Y(B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v,
                                           attn_mask=None,
                                           dropout_p=self.dropout if self.training else 0,
                                           is_causal=True)
        # Y(B, nh, T, hs) -> Y(B, T, nh, hs), contiguous()保证内存连续性, view()重塑形状为(B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Apply dropout and projection layer
        return self.att_dropout(self.att_proj(y))


# MLP
class FeedForward(nn.Module):
    def __init__(
            self,
            c: Config
    ):
        super().__init__()
        self.dropout = nn.Dropout(c.dropout)
        # 在LLaMA 3种MLP使用的是
        # down_proj(up(x) * SiLU(gate(x)))的结构，并且没有使用dropout
        self.up = nn.Linear(c.n_embed, 4 * c.n_embed, bias=c.bias)
        self.gate = nn.Linear(c.n_embed, 4 * c.n_embed, bias=c.bias)
        self.down_proj = nn.Linear(c.n_embed * 4, c.n_embed, bias=c.bias)

    def forward(self, x):
        output = self.down_proj(self.up(x) * F.silu(self.gate(x)))
        return self.dropout(output)


# BLOCK
class Block(nn.Module):
    def __init__(
            self,
            c: Config
    ):
        super().__init__()
        self.norm = RMSNorm(c.n_embed)
        self.attn = FlashAttention(c)
        self.mlp = FeedForward(c)

    def forward(self, x):
        # LLaMA 3中的Block结构
        # x  --> norm --> attn --> norm --> mlp---> y
        #     ↓-------x--------↑ ↓-------x-------↑
        x = x + self.attn(self.norm(x))
        return x + self.mlp(self.norm(x))


class GPT(nn.Module):
    def __init__(
            self,
            c: Config
    ):
        super().__init__()

        self.c = c
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embed),
            # 获取token_embed
            wpe=nn.Embedding(c.block_size, c.n_embed),
            # 使用一组可学习的位置编码pos_embed
            drop=nn.Dropout(c.dropout),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            norm=RMSNorm(c.n_embed)
        ))
        self.lm_head = nn.Linear(c.n_embed, c.vocab_size, bias=False)
        # share weights
        self.transformer.wte.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('_proj.weight'):  # 包括attn.att_proj以及mlp.down_proj
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / torch.sqrt(torch.tensor(c.n_layer, dtype=torch.float32)))

        print(f"Total parameters: {sum(p.numel() for p in self.parameters()) / 1_000_000} M")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # embedding
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embed)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embed)

        x = self.transformer.drop(tok_emb + pos_emb)  # combine token and position embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.norm(x)

        if target is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            # logits.size(-1) : vocab_size
            # logits.view(-1, logits.size(-1)) : (B*T, vocab_size)
            # target.view(-1) : (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        else:
            # B, T, C  -> B, 1, C -> B, C
            x = x[:, -1, :]
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # 对二维的参数使用weight decay
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # 其他不用
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        # 关于fused的解释：https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_gene_tokens, temperature=1.0, top_k=None):
        for _ in range(max_gene_tokens):
            # if idx.shape[1] > block_size ==> idx = idx[:, -block_size:]
            # idx.shape[1] <= block_size ==> idx = idx
            idx = idx if idx.shape[1] <= self.c.block_size else idx[:, -self.c.block_size:]
            logits, _ = self(idx)
            # temperature 主要是用来控制输出的多样性，temperature越大，输出的多样性越大
            # 在迭代的过程中，如果temperature很大，那么logits中的值会很小，导致softmax之后的概率分布比较平坦
            # 这样就会有更多的可能性被选择
            logits = logits[:, -1, :] / temperature

            # top_k是用来控制输出的多样性，top_k越大，输出的多样性越大
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 从概率分布中采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # 迭代的过程中，将采样的结果拼接到idx后面
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
