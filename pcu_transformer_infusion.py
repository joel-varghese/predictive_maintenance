class Config:
  feature_cols = [
      'PCUSerialNumber','ActiveStartTime', 'InfusionTime', 'EqActiveTime', 'TimeSinceMR', 'Useful_Time'
  ]
  redundant_features = [
       'WO_WO#'
  ]
  lr = 1e-4
  hidden_dim = 256
  d_ff = 2048
  prediction_length = 50
  num_layers = 3
  nhead = 4
  dropout = 0.05
  epochs = 20
  batch_size = 32
  seed = 42
  window_size = 250

import torch

class TriangularCausalMask():
    def __init__(self, B, L, device=torch.device('cuda')):
        mask_shape = [B,1,L,L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)

        self._mask = self._mask.to(device)

    @property
    def mask(self):
        return self._mask

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import random

def set_seed(seed: int = Config.seed) -> None:
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

set_seed()

import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd

class InfusionPumpDataset(Dataset):
    def __init__(self, data, flag='train', window_size=250, prediction_length=50,  target_col='Useful_Time', feature_cols=None,
                 batch_size=32, scaler=None):
        self.seq_len = window_size
        self.prediction_length = prediction_length
        self.scaler = StandardScaler()
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0,'val':1,'test':2}
        self.set_type = type_map[flag]

        self.feature_cols = list(data.columns)
        self.target_col = target_col
        self.data = preprocess_timestamp(data)
        self.grouped = self.data.groupby('PCUSerialNumber')

        self.external_scaler = scaler is not None
        self.scaler = scaler if scaler else StandardScaler()

        self.__read_data__()

    def __read_data__(self):
        self.index_map = []

        self.feature_cols.remove(self.target_col)
        # self.feature_cols.remove('PCUSerialNumber')
        self.feature_cols.remove('WO_WO#')
        self.feature_cols.remove('ActiveStartTime')
        filtered_df = [group.sort_values('ActiveStartTime') for _,group in self.grouped if len(group) >= self.seq_len +
                       self.prediction_length + 2]

        all_data = []
        global_index_offset = 0

        # df = pd.concat(filtered_df, axis=0).reset_index(drop=True)
        # batch_df = []
        for group_df in filtered_df:
            group_len = len(group_df)

            max_possible_samples = group_len - self.seq_len - self.prediction_length + 1
            if max_possible_samples < 3:
                continue

            num_train = int(max_possible_samples * 0.8)
            num_val = int(max_possible_samples * 0.1)
            num_test = max_possible_samples - num_train - num_val

            for i in range(max_possible_samples):
                global_index = global_index_offset + i
                if self.set_type == 0 and i < num_train:
                    self.index_map.append(global_index)
                elif self.set_type == 1 and num_train <= i < num_train + num_val:
                    self.index_map.append(global_index)
                elif self.set_type == 2 and i >= num_train + num_val:
                    self.index_map.append(global_index)

            all_data.append(group_df)
            global_index_offset += group_len

        df = pd.concat(all_data, axis=0).reset_index(drop=True)
        self.data_x = df[['ActiveStartTime']+self.feature_cols+[self.target_col]]
        self.scaler.fit(self.data_x)
        self.data_x = self.scaler.transform(self.data_x)

    def __getitem__(self, index):
        local_index = self.index_map[index]

        s_begin = local_index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.prediction_length

        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_x[r_begin:r_end, -1]

        return seq_x, seq_y, index

    def __len__(self):
        return len(self.index_map)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def preprocess_timestamp(df):

  df = df.copy()
  df['ActiveStartTime'] = pd.to_datetime(df['ActiveStartTime'])
  df['ActiveStartTime'] = df['ActiveStartTime'].astype('int64') / 1e9
  return df

class LearnedMask(nn.Module):
    def __init__(self, d_model, n_heads, proj_dim=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.proj_dim = proj_dim or (d_model // 4)
        self.q_proj = nn.Linear(d_model, self.proj_dim)
        self.k_proj = nn.Linear(d_model, self.proj_dim)
        self.activation = nn.Tanh()

    def forward(self, Q, K):
        # Q: (B, L, H, D), K: (B, S, H, D)
        B, L, H, D = Q.shape
        S = K.shape[1]

        Q_flat = Q.reshape(B * L * H, D)
        K_flat = K.reshape(B * S * H, D)

        q_proj = self.q_proj(Q_flat).reshape(B, L, H, -1)
        k_proj = self.k_proj(K_flat).reshape(B, S, H, -1)

        q_proj = q_proj.permute(0, 2, 1, 3)
        k_proj = k_proj.permute(0, 2, 1, 3)

        bias = torch.einsum('bhld,bhsd->bhls', q_proj, k_proj)

        return self.activation(bias)

from math import sqrt

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False,
                 learned_mask=None):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.learned_mask_layer = learned_mask

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, D = queries.shape
        _, S, _, _ = values.shape

        scale = self.scale or 1./sqrt(D)

        scores = torch.einsum("blhd,bshd->bhls",queries, keys)
        scores = scores * scale

        if self.learned_mask_layer is not None:
            learned_bias = self.learned_mask_layer(queries, keys)  # (B, H, L, S)
            scores += learned_bias

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill_(attn_mask.unsqueeze(1), float('-inf'))

        attn = torch.softmax(scores, dim = -1)
        attn = self.dropout(attn)

        context = torch.einsum("bhls,bshd->blhd", attn, values)


        return (context.contiguous(), attn)



class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, self.d_keys)
        keys = self.key_projection(keys).view(B, S, H, self.d_keys)
        values = self.value_projection(values).view(B, S, H, self.d_values)

        context, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = context.view(B,L,-1)

        return self.out_projection(out), attn

import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.05, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x,x,x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).tranpose(-1,1))

        return self.norm3(x+y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=2048, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x,x,x,
            attn_mask = attn_mask
        )
        x = x + self.dropout1(new_x)
        x = self.norm1(x)
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(y)
        x = self.norm2(x)

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=1000):
    super(PositionalEncoding, self).__init__()
    pe=torch.zeros(max_len, d_model)
    position=torch.arange(0,max_len, dtype=torch.float).unsqueeze(1)
    div_term=torch.exp(torch.arange(0,d_model,2).float()*(-np.log(10000.0)/d_model))
    pe[:,0::2]=torch.sin(position*div_term)
    pe[:,1::2]=torch.cos(position*div_term)
    pe=pe.unsqueeze(0)
    self.register_buffer('pe',pe)

  def forward(self,x):
    x = x + self.pe[:,:x.size(1),:]
    return x

def init_weights(module):
  if isinstance(module, nn.Linear):
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
      nn.init.zeros_(module.bias)
  elif isinstance(module, nn.Embedding):
    nn.init.xavier_uniform_(module.weight)

class PCUTransformer(nn.Module):
  def __init__(self, feature_size=7, hidden_dim=256, d_ff=125, num_layers=2, nheads=4, dropout=0.1, prediction_length=50, activation='gelu'):
    super(PCUTransformer, self).__init__()
    self.feature_size = feature_size
    self.hidden_dim = hidden_dim
    self.prediction_length = prediction_length

    self.embedding = nn.Linear(feature_size, hidden_dim)
    output_attention = False
    self.pos_encoder = PositionalEncoding(hidden_dim)
    self.learned_mask = LearnedMask(d_model=hidden_dim // nheads, n_heads=nheads)
    Attn = FullAttention

    self.encoder = Encoder(
        [
            EncoderLayer(
                AttentionLayer(
                    attention=Attn(mask_flag=False,
                                   attention_dropout=dropout,
                                   output_attention=output_attention,
                                   learned_mask=self.learned_mask),
                    d_model=hidden_dim,
                    n_heads=nheads,
                    mix=False),
                d_model=hidden_dim,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            ) for l in range(num_layers)
        ],
        norm_layer = torch.nn.LayerNorm(hidden_dim)
    )
    self.fc = nn.Linear(hidden_dim, prediction_length)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    batch_size, seq_len, _ = x.size()
    # print("Shape of x is", x.shape)
    x = self.embedding(x)
    x = self.pos_encoder(x)
    # x = self.transformer_encoder(x)
    x, _ = self.encoder(x, attn_mask=None)
    # print("After encoding x is", x.shape)
    x = x[:,-1,:]
    x=self.dropout(x)
    x=self.fc(x)
    return x

def evaluation(y_test, y_pred):
  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  return mae, rmse

class TimeWeightedMSELoss(nn.Module):
    def __init__(self, prediction_length, gamma=0.9):
        super(TimeWeightedMSELoss, self).__init__()
        self.prediction_length = prediction_length
        self.weights = torch.tensor([gamma ** (prediction_length - i - 1) for i in range(prediction_length)],
                                    dtype=torch.float32)

    def forward(self, pred, target):
        if pred.device != self.weights.device:
            self.weights = self.weights.to(pred.device)
        loss = ((pred - target) ** 2) * self.weights
        return torch.mean(loss)

import time
from warnings import filterwarnings
filterwarnings("ignore")

class ModelBatcher:
  def __init__(self, data_df, lr, feature_size, hidden_dim, d_ff, num_layers, nheads,
               dropout, seed, window_size, prediction_length, batch_size, device):
    self.model = PCUTransformer(
        feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, nheads=nheads,
        dropout=dropout, prediction_length=prediction_length
    ).to(device).float()

    self.model.apply(init_weights)
    self.data = data_df
    self.window_size = window_size
    self.prediction_length = prediction_length
    # self.criterion = WeightedCustomLoss(alpha=1.0)
    # Here we replace with other loss function
    self.criterion = TimeWeightedMSELoss(prediction_length=prediction_length, gamma=0.9)
    self.batch_size = batch_size
    self.device = device
    self.lr = lr

  def _get_data_(self, flag, scaler):
    data_set = InfusionPumpDataset(data=self.data, flag = flag, window_size=self.window_size, prediction_length=self.prediction_length,
                                 target_col='Useful_Time', feature_cols=Config.feature_cols, batch_size=32, scaler=scaler)
    data_loader = DataLoader(data_set,
                            batch_size=self.batch_size,
                            pin_memory=True,
                            num_workers=4,
                            shuffle=False,
                            drop_last=True)

    return data_set, data_loader


  def inverse_transform_predictions(self, pred, truth, scaler, num_features, target_index):
    def reconstruct(data_slice):
      batch, pred_len = data_slice.shape
      full = np.zeros((batch*pred_len, num_features))
      full[:, target_index] = data_slice.flatten()
      return full

    pred_full = reconstruct(pred)
    truth_full = reconstruct(truth)

    pred_inv = scaler.inverse_transform(pred_full)[:, target_index]
    truth_inv = scaler.inverse_transform(truth_full)[:, target_index]

    return pred_inv, truth_inv


  def val_model(self, vali_data, vali_loader):
    mape_loss = []
    mse_loss = []
    cust_loss = []
    for batch_idx, (val_x, val_y, val_indices) in enumerate(vali_loader):
        with torch.no_grad():
            val_x = val_x.to(self.device).float()
            val_y = val_y.to(self.device).float()
            output = self.model(val_x)
            # print("Look at these op", output)
            # output = vali_data.inverse_transform(output.detach().cpu().numpy())
            loss = self.criterion(output, val_y)
            output_np = output.detach().cpu().numpy()
            val_y_np = val_y.detach().cpu().numpy()
            num_features = val_x.shape[2]
            target_index = num_features - 1
            output_inv, test_y_inv = self.inverse_transform_predictions(output_np, val_y_np, vali_data.scaler, num_features, target_index)
            mape = mean_absolute_percentage_error(output_np, val_y_np)
            mse = mean_squared_error(output_np, val_y_np)
            mse_loss.append(loss.item())
            mape_loss.append(mape)
            cust_loss.append(loss.item())
    mse_loss = np.average(mse_loss)
    mape_loss = np.average(mape_loss)
    cust_loss = np.average(cust_loss)
    self.model.train()

    return mse_loss, mape_loss, cust_loss

  def test_model(self, test_data, test_loader):
      mae_loss = []
      cust_loss = []
      mape_loss = []
      for batch_idx, (test_x, test_y, test_indices) in enumerate(test_loader):
          with torch.no_grad():
              test_x = test_x.to(self.device).float()
              test_y = test_y.to(self.device).float()
              output = self.model(test_x)
              # output = test_data.inverse_transform(output.detach().cpu().numpy())
              loss = self.criterion(output, test_y)
              output_np = output.detach().cpu().numpy()
              test_y_np = test_y.detach().cpu().numpy()
              mae = mean_absolute_error(output_np, test_y_np)
              mse = mean_squared_error(output_np, test_y_np)
              num_features = test_x.shape[2]
              target_index = num_features - 1
              output_inv, test_y_inv = self.inverse_transform_predictions(output_np, test_y_np, test_data.scaler, num_features, target_index)
              mape = mean_absolute_percentage_error(output_np, test_y_np)
              cust_loss.append(loss.item())
              mae_loss.append(mae)
              mape_loss.append(mape)
      mae_loss = np.average(mae_loss)
      cust_loss = np.average(cust_loss)
      mape_loss = np.average(mape_loss)

      return mae_loss, mape_loss, cust_loss

  def train_model(self, feature_size=8, hidden_dim=64, num_layers=2,
                  nheads=4, dropout=0.1, weight_decay=0.0, epochs=100, seed=42,
                  window_size=250, prediction_length=50, device='cpu', model_path='model_weights.pth'
                  ):
    set_seed()
    # train_x, train_y, test_x, test_y, scaler, feature_cols = get_train_test(
    #     self.data, window_size=window_size, prediction_length=prediction_length
    # )
    train_data , train_loader = self._get_data_('train', None)
    scaler = train_data.scaler
    val_data, val_loader = self._get_data_('val', scaler)
    test_data, test_loader = self._get_data_('test', scaler)

    device = torch.device('cuda')


    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    best_mae = float('inf')
    best_model_state = None
    train_weights = torch.ones(len(train_data), device=device, dtype=torch.float32)

    for epoch in range(epochs):
      self.model.train()
      iter_count = 0
      train_loss = []

      # output = self.model(train_x)
      # loss = criterion(output, train_y)
      epoch_time = time.time()
      for batch_idx, (batch_x, batch_y, batch_indices) in enumerate(train_loader):
          iter_count += 1
          self.model.zero_grad()
          batch_x = batch_x.to(self.device).float()
          batch_y = batch_y.to(self.device).float()
          # print("Shape of x", batch_x.shape)
          output = self.model(batch_x)
          # output = train_data.inverse_transform(output.detach().cpu().numpy())
          loss = self.criterion(output, batch_y)
          train_loss.append(loss.item())

          loss.backward()
          optimizer.step()

      self.model.eval()

      with torch.no_grad():
        mse_val, mape_val, cust_val = self.val_model(val_data, val_loader)
        mse_test, mape_test, cust_test = self.val_model(test_data, test_loader)

        mae_test, _, _ = self.test_model(test_data, test_loader)

        print(f"Epoch {epoch+1}, Val MAE: {mse_val:.6f}, Val MAPE: {mape_val:.6f}, Test MAE: {mse_test:.6f}, Test MAPE: {mape_test:.6f}")

        if mae_test < best_mae:
          best_mae = mae_test
          best_model_state = self.model.state_dict()
          torch.save(best_model_state, model_path)
          print(f"Saved best model weights to {model_path}")

    self.model.load_state_dict(best_model_state)

    self.model.eval()
    with torch.no_grad():
      mae, rmse, mape = self.test_model(test_data, test_loader)

    return mae, rmse, mape

if __name__ == "__main__":
  infusion_pumps = pd.read_csv("/content/sample_data/time_processed.csv")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


  trainerclass = ModelBatcher(
      data_df=infusion_pumps,
      lr=Config.lr,
      feature_size=6,
      hidden_dim=Config.hidden_dim,
      d_ff=Config.d_ff,
      num_layers=Config.num_layers,
      nheads=Config.nhead,
      dropout=Config.dropout,
      seed=Config.seed,
      window_size=Config.window_size,
      prediction_length=Config.prediction_length,
      batch_size=Config.batch_size,
      device=device
  )
  mae, rmse, mape  = trainerclass.train_model(feature_size=7, hidden_dim=64, num_layers=2,
                  nheads=4, dropout=0.1, weight_decay=1e-4, epochs=Config.epochs, seed=42,
                  window_size=250, prediction_length=Config.prediction_length, device=device, model_path='model_weights.pth')


  print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time
import torch

def plot_predictions(true_values, pred_values, num_sequences=5, title='True vs Predicted Useful_Time'):
    maes = [mean_absolute_error(true_values[i], pred_values[i]) for i in range(len(true_values))]
    k=5
    # k = len(true_values)

    top_k_indices = np.argsort(maes)[:k]


    for rank, idx in enumerate(top_k_indices):

        pred_mae = [mean_absolute_error(true_values[idx][seq], pred_values[idx][seq]) for seq in range(len(true_values[idx]))]

        top_one = np.argsort(pred_mae)[0]

        plt.figure(figsize=(12, 6))
        time_steps = np.arange(len(true_values[idx][top_one]))

        plt.plot(time_steps, true_values[idx][top_one], marker='o', linestyle='-', label=f'Sequence {idx+1} (True)', color='C0')
        plt.plot(time_steps, pred_values[idx][top_one], marker='x', linestyle='--', label=f'Sequence {idx+1} (Predicted)', color='C1', alpha=0.7)

        plt.xlabel('Time Step (Future Timestamps)')
        plt.ylabel('Future Useful Time')
        plt.title(f'{title}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        plt.show()
        # plt.savefig('Prediction_plot.png')

# For the model tester the original data is needed and the testing data can be seperate

class ModelTester:
  def __init__(self, model_path, data_df, window_size, prediction_length, batch_size, device):
    self.model_path = model_path
    self.data = data_df
    self.window_size = window_size
    self.prediction_length = prediction_length
    self.batch_size = batch_size
    self.device = device

  def _get_data_(self, flag, scaler):
    data_set = InfusionPumpDataset(data=self.data, flag = flag, window_size=self.window_size, prediction_length=self.prediction_length,
                                 target_col='Useful_Time', feature_cols=Config.feature_cols, batch_size=32, scaler=scaler)
    data_loader = DataLoader(data_set,
                            batch_size=self.batch_size,
                            pin_memory=True,
                            num_workers=4,
                            shuffle=False,
                            drop_last=True)

    return data_set, data_loader

  def _inverse_target_transform(self, scaler, dataset, target_index=-1):

    n_samples, pred_len = dataset.shape
    n_features = scaler.mean_.shape[0]

    dummy = np.zeros((n_samples * pred_len, n_features))
    dummy[:, target_index] = dataset.flatten()

    inv = scaler.inverse_transform(dummy)

    return inv[:, target_index].reshape(n_samples, pred_len)

  def predict_data(self, flag):
      device = torch.device('cuda')
      model = PCUTransformer(
          feature_size = len(Config.feature_cols),
          hidden_dim=Config.hidden_dim,
          num_layers=Config.num_layers,
          nheads=Config.nhead,
          dropout=Config.dropout,
          prediction_length=self.prediction_length
      ).to(device)

      state_dict = torch.load(self.model_path)
      print("Trained keys", state_dict.keys())
      print("Initial keys",model.state_dict().keys())
      model.load_state_dict(torch.load(self.model_path))
      model.eval()

      train_data, train_loader = self._get_data_('train', None)
      scaler = train_data.scaler
      val_data, val_loader = self._get_data_('val', scaler)

      trues = []
      preds = []

      for i, (batch_x, batch_y, batch_indices) in enumerate(val_loader):
        with torch.no_grad():
          batch_x = batch_x.to(device).float()
          batch_y = batch_y.to(device).float()
          output = model(batch_x)
          batch_y_np = batch_y.detach().cpu().numpy()
          output_np = output.detach().cpu().numpy()

          batch_y_inv = self._inverse_target_transform(val_data.scaler, batch_y_np)
          output_inv = self._inverse_target_transform(val_data.scaler, output_np)

          trues.append(batch_y_inv)
          preds.append(output_inv)
      print(len(trues))
      return trues, preds

infusion_pumps = pd.read_csv("/content/sample_data/time_processed.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

predictor = ModelTester(model_path='/content/sample_data/model_weights_inf.pth', data_df=infusion_pumps, window_size=250, prediction_length=Config.prediction_length, batch_size=32, device=device)

trues, preds = predictor.predict_data('val')


plot_predictions(trues, preds)

