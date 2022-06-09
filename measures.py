from contextlib import contextmanager
from copy import deepcopy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
import torch.nn as nn

from experiment_config import DatasetType
from experiment_config import ComplexityType as CT
from utils.utils_CKA import *
from utils.data_utils import get_masks_and_count_tokens, get_src_and_trg_batches
from utils.optimizers_and_distributions import LabelSmoothingDistribution

@torch.no_grad()
def eval_batch(batch, model, labels):

  input_ids = batch['input_ids'].cuda()
  attention_mask = batch['attention_mask'].cuda()
  outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
  return outputs

@torch.no_grad()
def eval_acc(model, eval_loader):

  model.eval()

  num = 0
  correct = 0

  for batch in eval_loader:
    labels = batch['labels'].cuda()
    outputs = eval_batch(batch, model, labels)
    predictions = outputs.logits.argmax(dim=-1)
    num += len(labels)
    correct += (labels==predictions).sum().item()
    
  assert num>0
  acc = correct/num
  print(f"Evaluate accuracy = {acc}.")
  return acc

@torch.no_grad()
def eval_NMT_loss(model, dataloader, pad_token_id=None, trg_vocab_size=0, NMT_maximum_samples = 10000):

  num_processed_samples = 0
  device = next(model.parameters()).device
  training_loss = 0
  loss_step = 0

  for _, token_ids_batch in enumerate(dataloader):

    src_token_ids_batch, trg_token_ids_batch_input, target = get_src_and_trg_batches(token_ids_batch)
    num_processed_samples += token_ids_batch.batch_size
    src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)
    logits = model(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask) 

    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    label_smoothing = LabelSmoothingDistribution(0, pad_token_id, trg_vocab_size, device) # Use label smoothing = 0 here
    smooth_target_distributions = label_smoothing(target)  # these are regular probabilities
    loss = kl_div_loss(logits, smooth_target_distributions)
    training_loss += loss.item()
    loss_step += 1

    if num_processed_samples>=NMT_maximum_samples:
      break
  
  training_loss = training_loss/loss_step
  print(f"NMT training loss is {training_loss}")

  return training_loss

class ExperimentBaseModel(nn.Module):
  def __init__(self, dataset_type: DatasetType):
    super().__init__()
    self.dataset_type = dataset_type

  def forward(self, x) -> Tensor:
    raise NotImplementedError

# Adapted from https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
@torch.no_grad()
def _reparam(model):
  def in_place_reparam(model, prev_layer=None):
    for child in model.children():
      prev_layer = in_place_reparam(child, prev_layer)
      if child._get_name() == 'Conv2d':
        prev_layer = child
      elif child._get_name() == 'BatchNorm2d':
        scale = child.weight / ((child.running_var + child.eps).sqrt())
        prev_layer.bias.copy_( child.bias  + ( scale * (prev_layer.bias - child.running_mean) ) )
        perm = list(reversed(range(prev_layer.weight.dim())))
        prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale ).permute(perm))
        child.bias.fill_(0)
        child.weight.fill_(1)
        child.running_mean.fill_(0)
        child.running_var.fill_(1)
    return prev_layer
  model = deepcopy(model)
  in_place_reparam(model)
  return model


@contextmanager
def _perturbed_model(
  model: ExperimentBaseModel,
  sigma: float,
  rng,
  magnitude_eps: Optional[float] = None
):
  device = next(model.parameters()).device
  if magnitude_eps is not None:
    noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in model.parameters()]
  else:
    noise = [torch.normal(0,sigma**2,p.shape, generator=rng).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model


# Adapted from https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
def _pacbayes_sigma(
  model: ExperimentBaseModel,
  dataloader: DataLoader,
  accuracy: float,
  seed: int,
  magnitude_eps: Optional[float] = None,
  search_depth: int = 15,
  montecarlo_samples: int = 10,
  accuracy_displacement: float = 0.1,
  displacement_tolerance: float = 1e-2,
  task_type: str = 'normal',
  pad_token_id = None,
  trg_vocab_size: int = 0,
  pacbayes_depth: int = 15,
  search_upper_limit: float = 0.2
) -> float:
  
  if task_type == 'NMT' and magnitude_eps:
    # This is a tricky case. It seems that using search_upper_limit=0.2 is not large enough
    search_upper_limit = 2

  lower, upper = 0, search_upper_limit
  sigma = 0.1

  BIG_NUMBER = 10348628753
  device = next(model.parameters()).device
  rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
  rng.manual_seed(BIG_NUMBER + seed)

  if not accuracy and task_type == 'NMT':
    # In this case, the training accuracy is hard to evaluate
    # So we use the training loss instead
    # It is training loss, but we still call it "accuracy" to follow the convention
    print("Evaluate training loss using the original model.")
    accuracy = eval_NMT_loss(model, dataloader, pad_token_id=pad_token_id, trg_vocab_size=trg_vocab_size)
    accuracy_displacement = 0.5
    displacement_tolerance = 0.05

  print(f"Start binary search for PAC-Bayes sigma.")
  for _ in range(search_depth):
    sigma = (lower + upper) / 2
    # If sigma > search_upper_limit - 0.01, most likely the search is stuck because upper limit is too small
    if sigma > search_upper_limit * 0.95:
      return search_upper_limit

    accuracy_samples = []
    print(f"Getting samples for current sigma.")
    for _ in range(montecarlo_samples):
      print(f"current sigma is {sigma}")
      with _perturbed_model(model, sigma, rng, magnitude_eps) as p_model:
        # The following code is replaced with a method of evaluating accuracy
        #loss_estimate = 0
        #for data, target in dataloader:
        #  logits = p_model(data)
        #  pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
        #  batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
        #  loss_estimate += batch_correct.sum()
        #loss_estimate /= len(dataloader.dataset)
        if task_type == 'NMT':
          loss_estimate = eval_NMT_loss(p_model, dataloader, pad_token_id=pad_token_id, trg_vocab_size=trg_vocab_size)
        else:
          loss_estimate = eval_acc(p_model, dataloader)
        accuracy_samples.append(loss_estimate)
    displacement = abs(np.mean(accuracy_samples) - accuracy)
    if abs(displacement - accuracy_displacement) < displacement_tolerance:
      break
    elif displacement > accuracy_displacement:
      # Too much perturbation
      upper = sigma
    else:
      # Not perturbed enough to reach target displacement
      lower = sigma
  return sigma

def W_CKA(p,q, feature_space=True):

  eps=1e-15
  p = p.data.numpy()
  q = q.data.numpy()
  if np.sqrt(np.sum((p-q)**2)) < eps:
    return 1.0
  if feature_space:
    return feature_space_linear_cka(p, q)
  else:
    return cka_compute(gram_linear(p, q))

@torch.no_grad()
def get_all_measures(
  model: ExperimentBaseModel,
  init_model: ExperimentBaseModel,
  dataloader: DataLoader,
  acc: float,
  seed: int,
  no_path_norm=True,
  no_exact_spectral_norm=True,
  no_pac_bayes=False,
  no_margin=False,
  no_basics=False,
  no_CKA=True,
  task_type='NMT',
  path_norm_transformer=None,
  pad_token_id=None,
  trg_vocab_size=0,
  pacbayes_depth=15
) -> Dict[CT, float]:
  measures = {}

  model = _reparam(model)
  init_model = _reparam(init_model)

  device = next(model.parameters()).device
  m = len(dataloader.dataset)

  def get_weights_only(model: ExperimentBaseModel) -> List[Tensor]:
    blacklist = {'bias', 'bn'}
    return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]
  
  weights = get_weights_only(model)
  init_weights = get_weights_only(init_model)
  weights_cpu = [p.to("cpu") for p in weights]
  init_weights_cpu = [p.to("cpu") for p in init_weights]
  dist_init_weights = [p-q for p,q in zip(weights, init_weights)]
  d = len(weights)

  def get_vec_params(weights: List[Tensor]) -> Tensor:
    return torch.cat([p.view(-1) for p in weights], dim=0)

  w_vec = get_vec_params(weights)
  dist_w_vec = get_vec_params(dist_init_weights)
  num_params = len(w_vec)

  if not no_CKA:
    measures["W_CKA"] = np.mean([W_CKA(p,q, feature_space=True) for p,q in zip(weights_cpu, init_weights_cpu) if len(p.shape)>1])

  def get_reshaped_weights(weights: List[Tensor]) -> List[Tensor]:
    # If the weight is a tensor (e.g. a 4D Conv2d weight), it will be reshaped to a 2D matrix
    return [p.view(p.shape[0],-1) for p in weights]
  
  reshaped_weights = get_reshaped_weights(weights)
  dist_reshaped_weights = get_reshaped_weights(dist_init_weights)
  
  if not no_basics:
    print("Vector Norm Measures")
    measures["L2"] = w_vec.norm(p=2)
    measures["L2_DIST"] = dist_w_vec.norm(p=2)
    
    print("VC-Dimension Based Measures")
    measures["PARAMS"] = torch.tensor(num_params) # 20

  if not no_margin:
    print("Measures on the output of the network")
    def _calculate_margin(
      logits,
      target
    ):
      correct_logit = logits[torch.arange(logits.shape[0]), target].clone()
      logits[torch.arange(logits.shape[0]), target] = float('-inf')
      max_other_logit = logits.data.max(1).values  # get the index of the max logits
      margin = correct_logit - max_other_logit
      return margin

    @torch.no_grad()
    def _margin(
      model: ExperimentBaseModel,
      dataloader: DataLoader,
      task_type: str = 'normal',
      pad_token_id=None,
      NMT_maximum_samples = 10000,
    ) -> Tensor:
      margins = []
      if task_type=='NMT':
        num_processed_samples = 0
        for batch_id, token_ids_batch in enumerate(dataloader):
          src_token_ids_batch, trg_token_ids_batch_input, target = get_src_and_trg_batches(token_ids_batch)
          num_processed_samples += token_ids_batch.batch_size
          src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)
          logits = model(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask, no_softmax=True) # do not use softmax
          # The following is for calculating training loss
          #logits = model(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask, no_softmax=False) # do not use softmax
          margins.append(_calculate_margin(logits.clone(),target.flatten()))

          # The following is used for assessing the training loss
          #kl_div_loss = nn.KLDivLoss(reduction='batchmean')
          #BASELINE_MODEL_LABEL_SMOOTHING_VALUE = 0.1
          #trg_vocab_size = 7159
          #label_smoothing = LabelSmoothingDistribution(BASELINE_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, device)
          #smooth_target_distributions = label_smoothing(target)  # these are regular probabilities
          #loss = kl_div_loss(logits, smooth_target_distributions)
          #print(f"Training loss is {loss.item()}")
          
          if num_processed_samples >= NMT_maximum_samples:
            print(f"There are {num_processed_samples} sentences processed when calculating the margin.")
            break
        
        margin_distribution = torch.cat(margins)
        return margin_distribution.kthvalue(len(margin_distribution) // 10)[0]

      else:
        for batch in dataloader:
          target = batch['labels'].cuda()
          outputs = eval_batch(batch, model, target)
          logits = outputs.logits
          margins.append(_calculate_margin(logits,target))
      
        return torch.cat(margins).kthvalue(m // 10)[0]

    true_margin = _margin(model, dataloader, task_type, pad_token_id)
    measures["TRUE_MARGIN"] = true_margin # This measure is used to check if the true margin could become negative
    margin = true_margin.abs()
    measures["INVERSE_MARGIN"] = torch.tensor(1, device=device) / margin ** 2 # 22

  if not no_basics:
    print("(Norm & Margin)-Based Measures")
    fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in reshaped_weights])
    print("Starting SVD calculations which may occupy large memory.")
    spec_norms = torch.cat([p.svd().S.max().unsqueeze(0) ** 2 for p in reshaped_weights])
    print("End SVD calculations.")
    dist_fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in dist_reshaped_weights])
    dist_spec_norms = torch.cat([p.svd().S.max().unsqueeze(0) ** 2 for p in dist_reshaped_weights])

    print("Approximate Spectral Norm")
    # Note that these use an approximation from [Yoshida and Miyato, 2017]
    # https://arxiv.org/abs/1705.10941 (Section 3.2, Convolutions)
    measures["LOG_PROD_OF_SPEC"] = spec_norms.log().sum() # 32
    measures["FRO_OVER_SPEC"] = (fro_norms / spec_norms).sum() # 33
    measures["LOG_SUM_OF_SPEC"] = math.log(d) + (1/d) * measures["LOG_PROD_OF_SPEC"] # 35

  if not no_margin:
    measures["LOG_PROD_OF_SPEC_OVER_MARGIN"] = measures["LOG_PROD_OF_SPEC"] - 2 * margin.log() # 31
    measures["LOG_SPEC_INIT_MAIN"] = measures["LOG_PROD_OF_SPEC_OVER_MARGIN"] + (dist_fro_norms / spec_norms).sum().log() # 29
    measures["LOG_SPEC_ORIG_MAIN"] = measures["LOG_PROD_OF_SPEC_OVER_MARGIN"] + measures["FRO_OVER_SPEC"].log() # 30
    measures["LOG_SUM_OF_SPEC_OVER_MARGIN"] = math.log(d) + (1/d) * (measures["LOG_PROD_OF_SPEC"] -  2 * margin.log()) # 34
  
  if not no_basics:
    print("Frobenius Norm")
    measures["LOG_PROD_OF_FRO"] = fro_norms.log().sum() # 37
    measures["LOG_SUM_OF_FRO"] = math.log(d) + (1/d) * measures["LOG_PROD_OF_FRO"] # 39
    if not no_margin:
      measures["LOG_PROD_OF_FRO_OVER_MARGIN"] = measures["LOG_PROD_OF_FRO"] -  2 * margin.log() # 36
      measures["LOG_SUM_OF_FRO_OVER_MARGIN"] = math.log(d) + (1/d) * (measures["LOG_PROD_OF_FRO"] -  2 * margin.log()) # 38

    print("Distance to Initialization")
    measures["FRO_DIST"] = dist_fro_norms.sum() # 40
    measures["DIST_SPEC_INIT"] = dist_spec_norms.sum() # 41
    measures["PARAM_NORM"] = fro_norms.sum() # 42
  
  if not no_path_norm:
    print("Path-norm")
    # Adapted from https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py#L98
    def _path_norm(model: ExperimentBaseModel) -> Tensor:
      model = deepcopy(model)
      model.eval()
      for param in model.parameters():
        if param.requires_grad:
          param.data.pow_(2)
      # path norm requires all 1 input
      # we construct the all 1 input using length-1 sequence      
      model.src_embedding.embeddings_table.weight.data = torch.ones_like(model.src_embedding.embeddings_table.weight.data)
      model.src_pos_embedding.positional_encodings_table.data = torch.zeros_like(model.src_pos_embedding.positional_encodings_table.data)
      model.trg_embedding.embeddings_table.weight.data = torch.ones_like(model.trg_embedding.embeddings_table.weight.data)
      model.trg_pos_embedding.positional_encodings_table.data = torch.zeros_like(model.trg_pos_embedding.positional_encodings_table.data)
      
      if task_type == 'NMT':
        src_token=torch.ones(1,1).long()
        trg_token=torch.ones(1,1).long()
        src_mask=torch.ones(1,1,1,1)>0
        trg_mask=torch.ones(1,1,1,1)>0
        x = model(src_token, trg_token, src_mask, trg_mask)
      else:
        raise ValueError
      #x = torch.ones([1] + list(model.dataset_type.D), device=device)
      #x = model(x)
      del model
      return x.sum()
  
    measures["PATH_NORM"] = _path_norm(path_norm_transformer) # 44
    if not no_margin:
      measures["PATH_NORM_OVER_MARGIN"] = measures["PATH_NORM"] / margin ** 2 # 43
  
  if not no_exact_spectral_norm:
    print("Exact Spectral Norm")
    # Proposed in https://arxiv.org/abs/1805.10408
    # Adapted from https://github.com/brain-research/conv-sv/blob/master/conv2d_singular_values.py#L52
    def _spectral_norm_fft(kernel: Tensor, input_shape: Tuple[int, int]) -> Tensor:
      # PyTorch conv2d filters use Shape(out,in,kh,kw)
      # [Sedghi 2018] code expects filters of Shape(kh,kw,in,out)
      # Pytorch doesn't support complex FFT and SVD, so we do this in numpy
      np_kernel = np.einsum('oihw->hwio', kernel.data.cpu().numpy())
      transforms = np.fft.fft2(np_kernel, input_shape, axes=[0, 1]) # Shape(ih,iw,in,out)
      singular_values = np.linalg.svd(transforms, compute_uv=False) # Shape(ih,iw,min(in,out))
      spec_norm = singular_values.max()
      return torch.tensor(spec_norm, device=kernel.device)
  
    input_shape = (model.dataset_type.D[1], model.dataset_type.D[2])
    fft_spec_norms = torch.cat([_spectral_norm_fft(p, input_shape).unsqueeze(0) ** 2 for p in weights])
    fft_dist_spec_norms = torch.cat([_spectral_norm_fft(p, input_shape).unsqueeze(0) ** 2 for p in dist_init_weights])
  
    measures[CT.LOG_PROD_OF_SPEC_FFT] = fft_spec_norms.log().sum() # 32
    measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN_FFT] = measures[CT.LOG_PROD_OF_SPEC_FFT] - 2 * margin.log() # 31
    measures[CT.FRO_OVER_SPEC_FFT] = (fro_norms / fft_spec_norms).sum() # 33
    measures[CT.LOG_SUM_OF_SPEC_OVER_MARGIN_FFT] = math.log(d) + (1/d) * (measures[CT.LOG_PROD_OF_SPEC_FFT] -  2 * margin.log()) # 34
    measures[CT.LOG_SUM_OF_SPEC_FFT] = math.log(d) + (1/d) * measures[CT.LOG_PROD_OF_SPEC_FFT] # 35
    measures[CT.DIST_SPEC_INIT_FFT] = fft_dist_spec_norms.sum() # 41
    measures[CT.LOG_SPEC_INIT_MAIN_FFT] = measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN_FFT] + (dist_fro_norms / fft_spec_norms).sum().log() # 29
    measures[CT.LOG_SPEC_ORIG_MAIN_FFT] = measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN_FFT] + measures[CT.FRO_OVER_SPEC_FFT].log() # 30

  if not no_pac_bayes:
    print("Flatness-based measures")
    sigma = _pacbayes_sigma(model, dataloader, acc, seed, search_depth=pacbayes_depth, task_type=task_type, pad_token_id=pad_token_id, trg_vocab_size=trg_vocab_size)
    def _pacbayes_bound(reference_vec: Tensor) -> Tensor:
      return (reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2) + math.log(m / sigma) + 10
    measures["PACBAYES_INIT"] = _pacbayes_bound(dist_w_vec) # 48
    measures["PACBAYES_ORIG"] = _pacbayes_bound(w_vec) # 49
    measures["PACBAYES_FLATNESS"] = torch.tensor(1 / sigma ** 2) # 53
  
    print("Magnitude-aware Perturbation Bounds")
    mag_eps = 1e-3
    mag_sigma = _pacbayes_sigma(model, dataloader, acc, seed, mag_eps, search_depth=pacbayes_depth, task_type=task_type, pad_token_id=pad_token_id, trg_vocab_size=trg_vocab_size)
    omega = num_params
    def _pacbayes_mag_bound(reference_vec: Tensor) -> Tensor:
      numerator = mag_eps ** 2 + (mag_sigma ** 2 + 1) * (reference_vec.norm(p=2)**2) / omega
      denominator = mag_eps ** 2 + mag_sigma ** 2 * dist_w_vec ** 2
      return 1/4 * (numerator / denominator).log().sum() + math.log(m / mag_sigma) + 10
    measures["PACBAYES_MAG_INIT"] = _pacbayes_mag_bound(dist_w_vec) # 56
    measures["PACBAYES_MAG_ORIG"] = _pacbayes_mag_bound(w_vec) # 57
    measures["PACBAYES_MAG_FLATNESS"] = torch.tensor(1 / mag_sigma ** 2) # 61

  # Adjust for dataset size
  def adjust_measure(measure: CT, value: float) -> float:
    #if measure.name.startswith('LOG_'):
    if measure.startswith('LOG_'):
      return 0.5 * (value - np.log(m))
    elif 'CKA' in measure or 'TRUE_MARGIN' in measure:
      return value
    else:
      return np.sqrt(value / m)
  return {k: adjust_measure(k, v.item()) for k, v in measures.items()}