import torch
import torch.nn as nn
import torch.nn.functional as F


class DKVMN(nn.Module):
    """
    Minimal DKVMN:
    - Key memory: M x d_k (learned)
    - Value memory: M x d_v (dynamic per sequence; initialized from learned init)
    - Skill embedding: n_skills x d_k
    - Interaction embedding: n_skills*2 x d_v (skill + correctness)
    - Read -> predict p(correct)
    - Write with erase/add
    """

    def __init__(self, n_skills: int, memory_size: int = 50, d_k: int = 64, d_v: int = 64):
        super().__init__()
        self.n_skills = n_skills
        self.M = memory_size
        self.d_k = d_k
        self.d_v = d_v

        # Skill -> key space
        self.skill_embed = nn.Embedding(n_skills, d_k)

        # Keys (concept slots)
        self.key_memory = nn.Parameter(torch.randn(memory_size, d_k) * 0.1)

        # Initial value memory (learned starting state)
        self.value_init = nn.Parameter(torch.zeros(memory_size, d_v))

        # Interaction embedding: (skill, correct) -> value-space vector
        # index = skill_id + correct*n_skills
        self.interaction_embed = nn.Embedding(n_skills * 2, d_v)

        # Write networks
        self.erase = nn.Linear(d_v, d_v)
        self.add = nn.Linear(d_v, d_v)

        # Prediction network
        self.fc1 = nn.Linear(d_k + d_v, 128)
        self.fc2 = nn.Linear(128, 1)

    def _attention(self, skill_ids: torch.Tensor) -> torch.Tensor:
        """
        skill_ids: (B,)
        returns attention weights over memory slots: (B, M)
        """
        q = self.skill_embed(skill_ids)                 # (B, d_k)
        # similarity with key memory: (B, M)
        sim = torch.matmul(q, self.key_memory.t())
        w = F.softmax(sim, dim=1)
        return w

    def forward(self, skill_seq: torch.Tensor, correct_seq: torch.Tensor, mask: torch.Tensor):
        """
        skill_seq: (B, T) long
        correct_seq: (B, T) long {0,1}
        mask: (B, T) float {0,1}
        Returns: pred_probs (B, T), targets (B, T)
        """
        B, T = skill_seq.shape
        device = skill_seq.device

        # initialize value memory per batch
        value_memory = self.value_init.unsqueeze(0).repeat(B, 1, 1).to(device)  # (B, M, d_v)

        preds = []
        for t in range(T):
            s_t = skill_seq[:, t]            # (B,)
            c_t = correct_seq[:, t]          # (B,)

            w = self._attention(s_t)         # (B, M)

            # Read: weighted sum of value memory
            read_content = torch.bmm(w.unsqueeze(1), value_memory).squeeze(1)  # (B, d_v)
            q = self.skill_embed(s_t)                                           # (B, d_k)

            # Predict
            x = torch.cat([q, read_content], dim=1)                             # (B, d_k + d_v)
            h = F.relu(self.fc1(x))
            logit = self.fc2(h).squeeze(1)
            p = torch.sigmoid(logit)                                            # (B,)
            preds.append(p)

            # Write (update memory) using interaction embedding
            inter_idx = s_t + c_t * self.n_skills                               # (B,)
            v = self.interaction_embed(inter_idx)                               # (B, d_v)

            e = torch.sigmoid(self.erase(v))                                    # (B, d_v)
            a = torch.tanh(self.add(v))                                         # (B, d_v)

            # broadcast to memory slots with attention weights
            w_exp = w.unsqueeze(2)                                              # (B, M, 1)
            e_exp = e.unsqueeze(1)                                              # (B, 1, d_v)
            a_exp = a.unsqueeze(1)                                              # (B, 1, d_v)

            # value_memory = value_memory * (1 - w ⊗ e) + (w ⊗ a)
            value_memory = value_memory * (1.0 - w_exp * e_exp) + (w_exp * a_exp)

        pred_probs = torch.stack(preds, dim=1)                                  # (B, T)
        targets = correct_seq.float()
        return pred_probs * mask, targets * mask

    @torch.no_grad()
    def infer_mastery(self, skill_history: torch.Tensor, correct_history: torch.Tensor):
        """
        Run through a single student's history and return a mastery score per skill.
        mastery score = predicted P(correct) if we were to ask the student that skill now.

        skill_history: (T,) long
        correct_history: (T,) long
        returns mastery: (n_skills,) float in [0,1]
        """
        device = self.key_memory.device
        value_memory = self.value_init.clone().to(device)  # (M, d_v)

        T = skill_history.shape[0]
        for t in range(T):
            s_t = skill_history[t].view(1)     # (1,)
            c_t = correct_history[t].view(1)   # (1,)

            w = self._attention(s_t)           # (1, M)

            # write update
            inter_idx = s_t + c_t * self.n_skills
            v = self.interaction_embed(inter_idx)  # (1, d_v)
            e = torch.sigmoid(self.erase(v))       # (1, d_v)
            a = torch.tanh(self.add(v))            # (1, d_v)

            w_exp = w.squeeze(0).unsqueeze(1)      # (M,1)
            e_exp = e.squeeze(0).unsqueeze(0)      # (1,d_v)
            a_exp = a.squeeze(0).unsqueeze(0)      # (1,d_v)

            value_memory = value_memory * (1.0 - w_exp * e_exp) + (w_exp * a_exp)

        # Query mastery for each skill: compute p(correct) with current memory
        mastery = torch.zeros(self.n_skills, device=device)
        for sid in range(self.n_skills):
            s = torch.tensor([sid], device=device)
            w = self._attention(s)  # (1,M)
            read_content = torch.matmul(w, value_memory)  # (1,d_v)
            q = self.skill_embed(s)  # (1,d_k)

            x = torch.cat([q, read_content], dim=1)
            h = F.relu(self.fc1(x))
            logit = self.fc2(h).squeeze(1)
            mastery[sid] = torch.sigmoid(logit)

        return mastery.detach().cpu()