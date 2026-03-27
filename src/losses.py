import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCSRCELoss(nn.Module):
    """
    Adaptive Cluster-Separation-Rate-based (CSR) Cross Entropy Loss Function Class
    """
    def __init__(self,
                 num_classes,
                 feat_dim,
                 margin=0.01,
                 temperature=0.1,
                 csr_lambda=8.0,
                 csr_theta=0.6,
                 csr_gamma=0.95,
                 ce_weights=None,
                 hard_negative=True):
        """
        Args:
            num_classes
            feat_dim
            margin
            temperature
            csr_lambda: sigmoid slope (usual 5-10)
            csr_theta: target separation level (use validation to determine)
            csr_gamma: exponential moving average (EMA) smoothering for CSR (0.9-0.99)
            ce_weights
            hard_negative: hard negative mining
        """
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.temp = temperature
        self.hard_negative = hard_negative
        if self.hard_negative: self.k_hard = 0.4
        self.eps = 1e-16

        self.register_buffer('margin', torch.tensor(margin))
        self.register_buffer('centers', F.normalize(torch.randn(num_classes, feat_dim), p=2, dim=1))
        self.register_buffer('csr_ema', torch.tensor(0.5)) # start from the neutral value
        self.register_buffer('csr_lambda', torch.tensor(csr_lambda))
        self.register_buffer('csr_theta', torch.tensor(csr_theta))
        self.register_buffer('csr_gamma', torch.tensor(csr_gamma))

        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=ce_weights)

    def _compute_csr(self, emb, labels):
        """
        Compute Cluster Separation Ratio (CSR) for the given batch
        Args:
            emb
            labels
        Returns:
            ... 
        """
        emb_norm = F.normalize(emb, p=2, dim=1)
        centers = self.centers[labels] # target centers for the given batch
        # Mean intra-cluster cosine distance
        intra_dist = 1.0 - torch.sum(emb_norm*centers, dim=1)
        mean_intra = intra_dist.mean()
        # Mean inter-cluster cosine distance (between the centers)
        with torch.no_grad():
            unique_labels = torch.unique(labels)
            batch_centers = self.centers[unique_labels]
            cos_dist_inter = 1.0 - torch.matmul(batch_centers, batch_centers.t())
            # get upper triangle without diagonal
            mask = torch.triu(torch.ones_like(cos_dist_inter), diagonal=1).bool()
            mean_inter = cos_dist_inter[mask].mean() if mask.sum() > 0 else torch.tensor(1.0)
        cluster_separation_rate = mean_inter / (mean_intra + self.eps)
        return cluster_separation_rate.clamp(0.0, 10.0)
    
    def _cosine_loss(self, emb, labels):

        cos_sims = torch.matmul(emb, emb.t()) / self.temp
        same_label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        positive_mask = same_label_mask.clone().fill_diagonal_(0)
        positive_pairs = cos_sims[positive_mask]
        loss_positive = (1.0 - positive_pairs).clamp(min=0).mean()
        
        negative_mask = ~same_label_mask
        negative_pairs = cos_sims[negative_mask]
        scaled_margin = self.margin / self.temp
        if self.hard_negative:
            active_negative_mask = negative_pairs > scaled_margin
            active_negative_pairs = negative_pairs[active_negative_mask]
            if active_negative_pairs.numel() > 0:
                k = max(1, int(self.k_hard*active_negative_pairs.numel()))
                hard_negative_pairs, _ = torch.topk(active_negative_pairs, k)
                loss_negative = (hard_negative_pairs - scaled_margin).clamp(min=0).mean()
            else:
                loss_negative = (negative_pairs - scaled_margin).clamp(min=0).mean()
        else:
            loss_negative = (negative_pairs - scaled_margin).clamp(min=0).mean()
        # Collect statistics
        with torch.no_grad():
            self.accum_dist_pos += (1.0 - positive_pairs.mean().detach() * self.temp)
            self.accum_dist_neg += (1.0 - negative_pairs.mean().detach() * self.temp)
        return (loss_positive + loss_negative).clamp(min=self.eps)
    
    def _center_loss(self, emb, labels):

        with torch.no_grad():
            batch_centers_sum = torch.zeros_like(self.centers)
            batch_centers_sum.scatter_add_(0, labels.unsqueeze(1).expand(-1, emb.size(1)), emb)

            counts = torch.bincount(labels, minlength=self.num_classes).unsqueeze(1).float()
            mask = counts > 0

            new_centers = batch_centers_sum / (counts + self.eps)
            self.centers[mask.squeeze()] = self.gamma * self.centers[mask.squeeze()] + (1 - self.gamma) * new_centers[mask.squeeze()]
            self.centers.copy_(F.normalize(self.centers, p=2, dim=1))
        
        target_centers = self.centers[labels]
        cos_sims = torch.sum(emb * target_centers, dim=1) / self.temp
        return torch.mean(1.0 - cos_sims).clamp(min=self.eps)
    
    def forward(self, emb, logits, labels):
        """
        Args:
            emb
            logits
            labels
        Returns:
            loss
        """
        emb_norm = F.normalize(emb, p=2, dim=1)
        loss_ce = self.cross_entropy_loss(logits / self.temp, labels)
        loss_cos = self._cosine_loss(emb_norm, labels)
        loss_center = self._center_loss(emb_norm, labels)
        loss_metric = loss_cos + loss_center
        # Compute CSR
        csr_feedback = self._compute_csr(emb_norm, labels)
        # Compute EMA smoothing for CSR
        self.csr_ema = self.csr_gamma * self.csr_ema + (1 - self.csr_gamma) * csr_feedback
        # Compute multitask coefficients
        alpha = 1.0 / (1.0 + torch.exp(self.csr_lambda*(self.csr_ema - self.csr_theta)))
        beta = 1.0 - alpha
        return beta * loss_ce + alpha * loss_metric


class AdaptiveCosineCenterCrossEntropyLoss(nn.Module):
    """
    Adaptive Cosine Center Cross Entropy Loss Function Class
    """
    def __init__(self,
                 num_classes,
                 feat_dim,
                 ce_weights=None,
                 margin=0.01,
                 alpha=0.1,
                 beta=0.01,
                 gamma=0.9,
                 temperature=0.1,
                 hard_negative=True):
        """
        Args:
            num_classes
            feat_dim
            ce_weights
            margin
            alpha
            beta
            gamma
            temperature
            hard_negative
        """
        super(AdaptiveCosineCenterCrossEntropyLoss, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.gamma = gamma
        self.temp = temperature
        self.hard_negative = hard_negative
        if self.hard_negative: k_hard = 0.4
        self.eps = 1e-16

        self.register_buffer('centers', F.normalize(torch.randn(num_classes, feat_dim), p=2, dim=1))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))
        self.register_buffer('margin', torch.tensor(margin))
        self.register_buffer('k_hard', torch.tensor(k_hard))

        self.register_buffer('accum_dist_pos', torch.tensor(0.0))
        self.register_buffer('accum_dist_neg', torch.tensor(0.0))
        self.register_buffer('step_count', torch.tensor(0.0))
        self.register_buffer('accum_acc', torch.tensor(0.0))

        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=ce_weights)

    def update_params(self):

        msg = "Skip loss parameter change"
        if self.step_count.item() > 0:
            avg_pos = self.accum_dist_pos / self.step_count
            avg_neg = self.accum_dist_neg / self.step_count
            avg_acc = self.accum_acc / self.step_count

            sep_ratio = avg_pos / (avg_neg + self.eps)
            if sep_ratio > 0.3:
                self.alpha.sub_(0.005)
                self.beta.add_(0.005)
                msg = "Сlusters are too sparse - decrease alpha and increase beta"
            elif sep_ratio < 0.1:
                self.alpha.add_(0.005)
                self.beta.sub_(0.005)
                msg = "Сlusters are too dense - increase alpha and decrease beta"

            self.beta.clamp_(0.05, 0.5)
            self.alpha.clamp_(0.05, 0.5)

            new_margin = 0.9 * self.margin + 0.1 * avg_neg
            self.margin.copy_(torch.clamp(new_margin, max=0.4))

            if self.hard_negative: self.k_hard.copy_(torch.tensor(max(0.05, 0.2 * (1.0 - avg_acc.item()))))
            # Reset
            for res in [self.accum_dist_pos, self.accum_dist_neg, self.step_count, self.accum_acc]:
                res.fill_(0)

        return self.alpha, self.beta, self.margin, self.k_hard, msg

    def _cosine_loss(self, emb, labels):

        cos_sims = torch.matmul(emb, emb.t()) / self.temp

        same_label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_positive = same_label_mask.clone().fill_diagonal_(0)
        mask_negative = ~same_label_mask

        positive_pairs = cos_sims[mask_positive]
        negative_pairs = cos_sims[mask_negative]

        loss_positive = (1.0 - positive_pairs).clamp(min=0).mean()

        scaled_margin = self.margin / self.temp
        if self.hard_negative:
            active_mask_negative = negative_pairs > scaled_margin # 0
            active_negative_pairs = negative_pairs[active_mask_negative]
            if active_negative_pairs.numel() > 0:
                k = max(1, int(self.k_hard * active_negative_pairs.numel()))
                hard_negative_pairs, _ = torch.topk(active_negative_pairs, k)
                loss_negative = (hard_negative_pairs - scaled_margin).clamp(min=0).mean()
            else:
                loss_negative = (negative_pairs - scaled_margin).clamp(min=0).mean()
        else:
            loss_negative = (negative_pairs - scaled_margin).clamp(min=0).mean()
        # Collect statistics
        with torch.no_grad():
            self.accum_dist_pos += (1.0 - positive_pairs.mean().detach() * self.temp)
            self.accum_dist_neg += (1.0 - negative_pairs.mean().detach() * self.temp)

        return (loss_positive + loss_negative).clamp(min=self.eps)

    def _center_loss(self, emb, labels):

        with torch.no_grad():
            batch_centers_sum = torch.zeros_like(self.centers)
            batch_centers_sum.scatter_add_(0, labels.unsqueeze(1).expand(-1, emb.size(1)), emb)

            counts = torch.bincount(labels, minlength=self.num_classes).unsqueeze(1).float()
            mask = counts > 0

            new_centers = batch_centers_sum / (counts + self.eps)
            self.centers[mask.squeeze()] = self.gamma * self.centers[mask.squeeze()] + (1 - self.gamma) * new_centers[mask.squeeze()]
            self.centers.copy_(F.normalize(self.centers, p=2, dim=1))

        target_centers = self.centers[labels]
        cosine_sim = torch.sum(emb * target_centers, dim=1) / self.temp

        return torch.mean(1.0 - cosine_sim).clamp(min=self.eps)

    def forward(self, emb, logits, labels):
        """
        Args:
            emb
            logit
            labels
        Returns:
            ...
        """
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            self.accum_acc += (pred == labels).float().mean()
            self.step_count += 1
        emb_norm = F.normalize(emb, p=2, dim=1)
        loss_ce = self.cross_entropy_loss(logits / self.temp, labels)
        loss_cos = self._cosine_loss(emb_norm, labels)
        loss_center = self._center_loss(emb_norm, labels)

        return loss_ce + self.alpha * loss_cos + self.beta * loss_center


class CosineCenterCrossEntropyLoss(nn.Module):

    def __init__(self, ce_weights=None, margin=0.2, alpha=0.4, beta=0.1, hard_negative=True):
        super(CosineCenterCrossEntropyLoss, self).__init__()
        if hard_negative: # Hard Negative Mining (HNM)
            self.hard_negative = hard_negative
            self.k_hard = 0.2
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        if ce_weights is not None:
            self.cross_entropy_loss = nn.CrossEntropyLoss(weight=ce_weights)
        else:
            self.cross_entropy_loss = nn.CrossEntropyLoss()

    def _cosine_loss(self, emb, labels):

        cos_sims = torch.matmul(emb, emb.t())
        same_label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_positive = same_label_mask.clone().fill_diagonal_(0)
        mask_negative = ~same_label_mask

        positive_pairs = cos_sims[mask_positive]
        negative_pairs = cos_sims[mask_negative]

        loss_positive = (1.0 - positive_pairs).clamp(min=0).mean()

        if self.hard_negative:
            active_mask_negative = negative_pairs > self.margin # 0
            active_negative_pairs = negative_pairs[active_mask_negative]
            if active_negative_pairs.numel() > 0:
                k = max(1, int(self.k_hard * active_negative_pairs.numel()))
                hard_negative_pairs, _ = torch.topk(active_negative_pairs, k)
                loss_negative = (hard_negative_pairs - self.margin).clamp(min=0).mean()
            else:
                loss_negative = (negative_pairs - self.margin).clamp(min=0).mean()
        else:
            loss_negative = (negative_pairs - self.margin).clamp(min=0).mean()

        # Compute the adaptive coeffs
        # beta center loss через отношение среднего расстояния внутри класса и между классами.
        # alpha cosine loss через динамику изменения beta
        with torch.no_grad():
            dist_pos = 1.0 - positive_pairs.mean()
            dist_neg = 1.0 - negative_pairs.mean()
            sep_ratio = dist_pos / (dist_neg + 1e-8)
            if sep_ratio > 0.5: # кластеры рыхлые
                self.beta += 0.01
            elif sep_ratio < 0.1: # кластеры избыточно плотные
                self.beta -= 0.01
            self.beta = max(0.1, min(self.beta, 0.5))
            self.alpha = 0.5 - self.beta
        # динамический порог margin на основе текущего среднего значения отрицательных пар
        self.margin = 0.9 * self.margin + 0.1 * dist_neg.item()

        return loss_positive + loss_negative

    def _center_loss(self, emb, labels):

        unique_labels, labels_count = labels.unique(return_counts=True)
        num_unique = unique_labels.size(0)
        batch_centers = torch.zeros(num_unique, emb.size(1), device=emb.device, dtype=emb.dtype)
        label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}
        mapped_labels = torch.tensor([label_mapping[l.item()] for l in labels], device=emb.device)
        mapped_labels_expanded = mapped_labels.unsqueeze(1).expand(-1, emb.size(1))
        batch_centers.scatter_add_(0, mapped_labels_expanded, emb)
        batch_centers /= labels_count.float().unsqueeze(1)
        batch_centers = F.normalize(batch_centers, p=2, dim=1)

        target_centers = batch_centers[mapped_labels]
        cosine_sim = torch.sum(emb * target_centers, dim=1)

        return torch.mean(1.0 - cosine_sim)

    def forward(self, emb, logits, labels):

        if self.hard_negative:
            # считаем адаптивный коэффициент жесткости
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean()
            self.k_hard = max(0.05, 0.2 * (1.0 - acc.item()))

        loss_ce = self.cross_entropy_loss(logits, labels)
        loss_cos = self._cosine_loss(emb, labels)
        loss_center = self._center_loss(emb, labels)

        return loss_ce + self.alpha * loss_cos + self.beta * loss_center


class CosCrossEntropyLoss(nn.Module):

    def __init__(self, margin=0.4, alpha=0.4, hard_negative=True):
        super().__init__()
        if hard_negative: # Hard Negative Mining (HNM)
            self.hard_negative = hard_negative
            self.k_hard = 0.3
        self.margin = margin
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, emb, logits, labels):

        cos_sims = torch.matmul(emb, emb.t())
        same_label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_positive = same_label_mask.clone().fill_diagonal_(0) # same_label_mask.fill_diagonal_(0)
        mask_negative = ~same_label_mask

        positive_pairs = cos_sims[mask_positive]
        negative_pairs = cos_sims[mask_negative]

        loss_positive = (1.0 - positive_pairs).clamp(min=0).mean()

        if self.hard_negative: # Hard Negative Mining (HNM)
            active_mask_negative = negative_pairs > self.margin # 0
            active_negative_pairs = negative_pairs[active_mask_negative]
            if active_negative_pairs.numel() > 0:
                k = max(1, int(self.k_hard * active_negative_pairs.numel()))
                hard_negative_pairs, _ = torch.topk(active_negative_pairs, k)
                loss_negative = (hard_negative_pairs - self.margin).clamp(min=0).mean()
            else:
                loss_negative = (negative_pairs - self.margin).clamp(min=0).mean()
        else:
            loss_negative = (negative_pairs - self.margin).clamp(min=0).mean()

        loss_metric = loss_positive + loss_negative
        loss_ce = self.cross_entropy(logits, labels)

        return loss_ce + self.alpha * loss_metric


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin=margin

    def forward(self, output, label):

        loss = 0
        distance_matrix = torch.cdist(output, output)
        for i in range(len(output)):
            for j in range(len(output)):
                if i != j:
                    if label[i] == label[j]:
                        loss += distance_matrix[i, j] ** 2
                    else:
                        loss += torch.clamp(self.margin - distance_matrix[i, j], min=0) ** 2
        loss /= (len(output) * (len(output) - 1))

        return loss
    