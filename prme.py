import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from utils import kmeans_l1

CONST = 0


class InferenceNetwork(nn.Module):
  def __init__(self, D_vocab, D_h, inference_layers):
    super(InferenceNetwork, self).__init__()
    self.D_vocab = D_vocab
    self.D_h = D_h
    block = []
    block.append(nn.Linear(self.D_vocab, inference_layers[0]))
    block.append(nn.ReLU())
    block.append(nn.BatchNorm1d(inference_layers[0]))
    for layer in range(len(inference_layers)-1):
      block.append(nn.Linear(inference_layers[layer], inference_layers[layer+1]))
      block.append(nn.ReLU())
      block.append(nn.BatchNorm1d(inference_layers[layer+1]))
    block.append(nn.Linear(inference_layers[-1], self.D_h))
    self.block = nn.Sequential(*block)

  def forward(self, x):
    out = self.block(x)
    return out


class DecoderNetwork(nn.Module):
  def __init__(self, D_ell, D_h, decoder_layers):
    super(DecoderNetwork, self).__init__()

    block_mean = []
    block_mean.append(nn.Linear(D_ell+D_h, decoder_layers[0]))
    block_mean.append(nn.ReLU())
    block_mean.append(nn.BatchNorm1d(decoder_layers[0]))
    for layer in range(len(decoder_layers)-1):
      block_mean.append(nn.Linear(decoder_layers[layer], decoder_layers[layer+1]))
      block_mean.append(nn.ReLU())
      block_mean.append(nn.BatchNorm1d(decoder_layers[layer+1]))
    block_mean.append(nn.Linear(decoder_layers[-1], 1))
    self.block_mean = nn.Sequential(*block_mean)

    block_logvar = []
    block_logvar.append(nn.Linear(D_ell+D_h, decoder_layers[0]))
    block_logvar.append(nn.ReLU())
    block_logvar.append(nn.BatchNorm1d(decoder_layers[0]))
    for layer in range(len(decoder_layers)-1):
      block_logvar.append(nn.Linear(decoder_layers[layer], decoder_layers[layer+1]))
      block_logvar.append(nn.ReLU())
      block_logvar.append(nn.BatchNorm1d(decoder_layers[layer+1]))
    block_logvar.append(nn.Linear(decoder_layers[-1], 1))
    self.block_logvar = nn.Sequential(*block_logvar)

  def forward(self, x):
    mean = self.block_mean(x)
    logvar = self.block_logvar(x)
    return mean, torch.exp(logvar)


class PRME(object):
  """PRME topic models"""

  def __init__(self, args):
    self.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    self.lr = args.learning_rate
    self.outer_iter = int(args.outer_iteration)
    self.inner_iter = int(args.inner_iteration)
    self.dataset = args.dataset

    self.K = args.num_topics
    self.D_vocab = args.vocab_size
    self.D_ell = args.ell_size
    self.D_h = args.h_size
    self.a0 = args.a0
    self.b0 = args.b0
    self.alpha0 = args.alpha0
    self.gamma0 = args.gamma0/self.D_vocab
    self.beta = args.beta0
    self.v = Variable(1/(1+self.alpha0)*torch.ones(
        self.K-1, device=self.device), requires_grad=True)
    self.ell = torch.randn(
        self.K, self.D_ell, requires_grad=True, device=self.device)
    self.gamma = self.gamma0*torch.ones(self.K, self.D_vocab,
        device=self.device) + .01*torch.randn(
        self.K, self.D_vocab, device=self.device)
    self.inference_layers = [int(
        item) for item in args.inference_layers.split(',')]
    self.inference_network = InferenceNetwork(
        self.D_vocab, self.D_h, self.inference_layers).to(self.device)
    self.decoder_layers = [int(
        item) for item in args.decoder_layers.split(',')]
    self.decoder_network = DecoderNetwork(
        self.D_ell, self.D_h, self.decoder_layers).to(self.device)
    self.network_reg = args.network_reg
    if args.vocab_filename is not None:
      vocab_data = np.load(args.vocab_filename)
      self.vocab = vocab_data['vocab']
    else:
      self.vocab = None

  def fit(self, data):
    """Fit the model.
    
    Args:
      data: a structured .npz object containing:
        - data['x_idx']: A list of numpy arrays showing unique words' indices
            in each document.
        - data['x_cnt']: A list of numpy arrays showing unique words' counts
            in each document.
    """
    x_idx = data['x_idx']
    x_cnt = data['x_cnt']
    for n in range(len(x_cnt)):
      x_cnt[n] = torch.FloatTensor(x_cnt[n]).to(self.device)
    # N is the total number of documents.
    N = len(x_idx)
    # M is a list of document unique word counts.
    M = [len(x_n_idx) for x_n_idx in x_idx]
    
    x_bow = torch.zeros(N, self.D_vocab, device=self.device)
    for n in range(N):
      x_bow[n, list(x_idx[n])] = x_cnt[n]
    # normalize the histogram
    x_bow = x_bow/torch.sum(x_bow, dim=1,
                            keepdim=True).repeat(1,self.D_vocab)
    phi = [torch.randn(self.K, M_n, device=self.device) for M_n in M]
    z_a = torch.ones(N, self.K, device=self.device)
    z_b = torch.ones(N, self.K, device=self.device)

    kmeans_centers = kmeans_l1(x_bow, self.K)
    self.gamma = N/self.K*kmeans_centers+.1*torch.randn(
        self.K, self.D_vocab, device=self.device) +1+self.gamma0
    self.ell = torch.mm(kmeans_centers, 
        torch.randn(self.D_vocab, self.D_ell, device=self.device))
    self.ell = self.ell/torch.norm(
        self.ell, dim=1, keepdim=True).repeat(1,self.D_ell)
    self.ell.requires_grad = True
    if self.vocab is not None:
      self.display_topics()

    # fit the model
    total_perplexity = []
    for iter_all in range(self.outer_iter):
      print(iter_all)
      phi = self.fit_phi(phi, M, x_idx, z_a, z_b)
      z_a, z_b = self.fit_z(z_a, z_b, x_bow, M, N, phi, x_cnt)
      self.fit_gamma(N, M, phi, x_idx, x_cnt, 0)
      self.fit_v_ell_inference_decoder(x_bow, N, z_a, z_b, 0)
    self.display_topics()
    self.display_topic_heatmaps(x_bow, 100)

  # individual parameters
  def fit_phi(self, phi, M, x_idx, z_a, z_b):
    for n, M_n in enumerate(M):
      E_ln_eta = torch.digamma(self.gamma[:,x_idx[n]]) - torch.digamma(
          torch.mm(torch.sum(self.gamma, dim=1, keepdim=True), torch.ones(
          1, M_n).to(self.device)))
      E_ln_z_n = torch.mm((torch.digamma(z_a[n,:]) + torch.log(
          z_b[n,:])).unsqueeze(1), torch.ones(1, M_n).to(self.device))
      phi_n = torch.exp(E_ln_eta+E_ln_z_n)/torch.mm(torch.ones(
          self.K, 1).to(self.device), torch.sum(
          torch.exp(E_ln_eta+E_ln_z_n), dim=0, keepdim=True))
      phi[n] = phi_n.data+1e-6
    return phi

  def fit_z(self, z_a, z_b, x_bow, M, N, phi, x_cnt):
    ln_p_k = torch.cat((
        torch.log(self.v),torch.ones(1).to(self.device)), 0) + torch.cat((
        torch.zeros(1).to(self.device),torch.cumsum(
        torch.log(1-self.v), dim=0)), 0)
    p_k = torch.exp(ln_p_k)

    h = self.inference_network(x_bow)
    hl = torch.cat((h.repeat(1,self.K).view(N*self.K, self.D_h),
        self.ell.repeat(N,1)), 1)
    decoder_output1, decoder_output2 = self.decoder_network(hl)
    decoder_mu_theta = decoder_output1.view(N, self.K)
    decoder_sigma2_theta = decoder_output2.view(N, self.K)
    E_exp_neg_theta = torch.exp(-decoder_mu_theta+decoder_sigma2_theta/2)
    for n, M_n in enumerate(M):
      sum_E_z_n = torch.dot(z_a[n,:], z_b[n,:])
      z_a[n,:] = self.beta*p_k.data + torch.sum(
          torch.mm(phi[n], torch.diag(x_cnt[n])), dim=1)
      z_b[n,:] = 1 / (
          E_exp_neg_theta[n,:].data + torch.sum(x_cnt[n])/sum_E_z_n)
    return z_a, z_b

  def fit_gamma(self, N_total, M, phi, x_idx, x_cnt, global_iter):
    """Closed-form update for gamma"""
    self.gamma = self.gamma0*torch.ones(self.K,self.D_vocab,device=self.device)
    for n, M_n in enumerate(M):
      self.gamma[:,x_idx[n]] += torch.mm(phi[n], torch.diag(x_cnt[n]))

  def fit_v_ell_inference_decoder(self, x_bow, N_total, z_a, z_b, global_iter):
    optimizer_v_ell_inference_decoder = optim.Adam([
        {'params': self.v},
        {'params': self.ell},
        {'params': self.inference_network.parameters()},
        {'params': self.decoder_network.parameters()}
        ], lr=self.lr)
    N = N_total
    network_iter = self.inner_iter
    prev_loss = 0   
    for iter_v_ell_inference_decoder in range(network_iter):
      optimizer_v_ell_inference_decoder.zero_grad()
      h = self.inference_network(x_bow)
      hl = torch.cat((h.repeat(1,self.K).view(N*self.K, self.D_h),
          self.ell.repeat(N,1)), 1)
      decoder_output1, decoder_output2 = self.decoder_network(hl)
      decoder_mu_theta = decoder_output1.view(N, self.K)
      decoder_sigma2_theta = decoder_output2.view(N, self.K)
      decoder_sigma2_theta.data.clamp_(min=1e-6, max=100)
      ln_p_v = (self.alpha0-1)*torch.sum(torch.log(1-self.v)) + CONST
      ln_p_k = torch.cat((
          torch.log(self.v),torch.ones(1).to(self.device)), 0) + torch.cat((
          torch.zeros(1).to(self.device),torch.cumsum(
          torch.log(1-self.v), dim=0)), 0)
      E_ln_z = torch.digamma(z_a) + torch.log(z_b)
      E_ln_p_z = -N*torch.sum(torch.lgamma(
        self.beta*torch.exp(ln_p_k))) - self.beta*torch.dot(torch.exp(ln_p_k),
        torch.sum(decoder_mu_theta-E_ln_z, dim=0)) - torch.sum(
        E_ln_z) - torch.sum(z_a*z_b*torch.exp(
        -decoder_mu_theta+decoder_sigma2_theta/2))
      E_ln_p_h = -N*self.D_h/2*torch.log(
          torch.tensor(2*np.pi*self.a0).to(self.device))-1/2/self.a0*torch.sum(
          h.pow(2))
      ln_p_ell = -self.K*self.D_ell/2*torch.log(
          2*np.pi*torch.tensor(self.b0).to(self.device)) - torch.norm(
          self.ell).pow(2)/2/self.b0
      network_norm = 0
      for param in self.inference_network.parameters():
        network_norm += torch.norm(param)
      for param in self.decoder_network.parameters():
        network_norm += torch.norm(param)
      net_norm = self.network_reg*network_norm
      loss = -ln_p_v-N_total/N*E_ln_p_z-N_total/N*E_ln_p_h-ln_p_ell+net_norm
      loss.backward()
      optimizer_v_ell_inference_decoder.step()
      self.v.data.clamp_(min=1e-6, max=1-1e-6)
      print(iter_v_ell_inference_decoder, loss)
      if torch.isnan(loss.data):
        print(h,
              ln_p_v,
              ln_p_k,
              E_ln_p_z,
              E_ln_p_h,
              ln_p_ell,
              network_norm)
        raise ValueError('Nan loss!')
      if (torch.abs((prev_loss-loss)/loss) <= 1e-6 and 
          iter_v_ell_inference_decoder>=50) or (iter_v_ell_inference_decoder
          == network_iter-1):
        break
      prev_loss = loss

  def local_bound(self, x_bow, N, M, z_a, z_b, phi, x_idx, x_cnt):
    h = self.inference_network(x_bow)
    hl = torch.cat((h.repeat(1,self.K).view(N*self.K, self.D_h),
          self.ell.detach().repeat(N,1)), 1)
    decoder_output1, decoder_output2 = self.decoder_network(hl)
    decoder_mu_theta = decoder_output1.view(N, self.K)
    decoder_sigma2_theta = decoder_output2.view(N, self.K)

    ln_p_v = (self.alpha0-1)*torch.sum(torch.log(1-self.v)) + CONST
    ln_p_k = torch.cat((
        torch.log(self.v),torch.ones(1).to(self.device)), 0) + torch.cat((
        torch.zeros(1).to(self.device),torch.cumsum(
        torch.log(1-self.v), dim=0)), 0)
    E_ln_z = torch.digamma(z_a) + torch.log(z_b)
    E_ln_eta = torch.digamma(self.gamma) - torch.digamma(
        torch.sum(self.gamma, dim=1, keepdim=True).repeat(1,self.D_vocab))
    
    E_ln_p_h = -N*self.D_h/2*torch.log(
        torch.tensor(2*np.pi*self.a0).to(self.device))-1/2/self.a0*torch.sum(
        h.pow(2))
    E_ln_p_z = -N*torch.sum(torch.lgamma(
        self.beta*torch.exp(ln_p_k))) - self.beta*torch.dot(
        torch.exp(ln_p_k), torch.sum(decoder_mu_theta-E_ln_z, dim=0)) - torch.sum(
        E_ln_z) - torch.sum(z_a*z_b*torch.exp(-decoder_mu_theta+decoder_sigma2_theta/2))
    E_ln_p_c = 0
    E_ln_p_x = 0

    H_z = torch.sum(
        z_a+torch.log(z_b)+torch.lgamma(z_a)+(1-z_a)*torch.digamma(z_a))
    H_c = 0

    for n, M_n in enumerate(M):
      sum_E_z_n = torch.dot(z_a[n,:], z_b[n,:])
      E_ln_p_c += torch.dot(torch.sum(torch.mm(phi[n],torch.diag(x_cnt[n])),
          dim=1), E_ln_z[n,:]) - torch.sum(x_cnt[n])*torch.log(sum_E_z_n)
      E_ln_p_x += torch.sum(
          torch.mm(phi[n]*E_ln_eta[:,x_idx[n]], torch.diag(x_cnt[n])))
      H_c -= torch.sum(
          torch.mm(phi[n]*torch.log(phi[n]), torch.diag(x_cnt[n])))

    l_local = (E_ln_p_h+E_ln_p_z+E_ln_p_c+E_ln_p_x) + (H_z+H_c)
    if torch.isnan(l_local.data):
      print(E_ln_p_h,
            E_ln_p_z,
            E_ln_p_c,
            E_ln_p_x,
            H_z,
            H_c)
      raise ValueError('Nan loss!')
    return l_local

  def global_bound(self):
    ln_p_ell = -self.D_ell*self.K/2*np.log(2*np.pi*self.b0) - torch.norm(
        self.ell.data).pow(2)/2/self.b0
    ln_p_v = (self.alpha0-1)*torch.sum(torch.log(1-self.v)) + (self.K-1)*(
        np.log(1+self.alpha0)-np.log(1)-np.log(self.alpha0))
    E_ln_eta = torch.digamma(self.gamma) - torch.digamma(
          torch.mm(torch.sum(self.gamma, dim=1, keepdim=True), torch.ones(
          1, self.D_vocab).to(self.device)))
    E_ln_p_eta = self.K*gammaln(
        self.D_vocab*self.gamma0) - self.K*self.D_vocab*gammaln(
        self.gamma0) + (self.gamma0-1)*torch.sum(E_ln_eta)
    H_eta =  - torch.sum(torch.lgamma(torch.sum(
        self.gamma, dim=1))) + torch.sum(
        torch.lgamma(self.gamma)) - torch.sum((self.gamma-1)*E_ln_eta)
    l_global = ln_p_ell.float().to(
        self.device)+ln_p_v.data+E_ln_p_eta.float().to(self.device)+H_eta
    return l_global

  def display_topics(self, top_n_words=8, top_n_similar_topics=5):
    for k in range(self.K):
      topn_words = torch.sort(self.gamma[k,:],
                              descending=True)[1][:top_n_words]
      topk_similar_topics = torch.sort(torch.norm(
                              self.ell[k:(k+1),:].repeat(self.K,1)-self.ell,
                              dim=1))[1][1:top_n_similar_topics+1]
      print('Topic{}: Most similar to topic {}'.format(
          k, topk_similar_topics.tolist()))
      print(self.vocab[topn_words])

  def display_topic_heatmaps(self, x_bow, nbin):
    ln_p_k = torch.cat((
        torch.log(self.v),torch.ones(1).to(self.device)), 0) + torch.cat((
        torch.zeros(1).to(self.device),torch.cumsum(
        torch.log(1-self.v), dim=0)), 0)
    h = self.inference_network(x_bow)
    N, _ = h.shape
    h_mean = torch.mean(h, dim=0, keepdim=True)
    h_centered =  h-h_mean.repeat(N,1)
    u_h, s_h, v_h = torch.svd(h_centered)
    disp_x_array = torch.linspace( -1, 1,
        steps=nbin, dtype=torch.float).unsqueeze(1).to(self.device)
    disp_y_array = torch.linspace( -1, 1,
        steps=nbin, dtype=torch.float).unsqueeze(1).to(self.device)
    disp_xy_array = torch.cat((disp_x_array.repeat(1,nbin).view(-1,1),
        disp_y_array.repeat(nbin,1)), 1)
    xy_array = torch.mm(torch.mm(disp_xy_array, torch.diag(s_h[:2])),
                        v_h[:2,:]) + h_mean.repeat(nbin*nbin,1)
    theta = torch.ones(nbin, nbin, self.K)
    fig = plt.figure(figsize=(10,8))
    for k in range(self.K):
      hl_k = torch.cat((
          xy_array, self.ell[k:k+1,:].repeat(nbin*nbin,1)), 1)
      decoder_output_k1, decoder_output_k2 = self.decoder_network(hl_k)
      theta[:,:,k] = decoder_output_k1.view(
                          nbin, nbin) + ln_p_k[k] + np.log(self.beta)
    E_Z = torch.exp(theta)
    sum_E_Z = torch.sum(E_Z, dim=2)
    for k in range(self.K):
       ax = fig.add_subplot(np.ceil(self.K/5), 5, k+1)
       ax.imshow((E_Z[:,:,k]/sum_E_Z).detach().cpu().numpy(), 
                  vmin=0, vmax=1, cmap='jet')
       top3_words = torch.sort(self.gamma[k,:], 
                               descending=True)[1][:3]
       plt.text(10, 93, str(k+1), color='white', 
                horizontalalignment='center', fontsize=16, weight='bold')
       plt.text(60, 73, self.vocab[top3_words][0], color='white', 
                horizontalalignment='center', fontsize=14, weight='bold')
       plt.text(60, 84, self.vocab[top3_words][1], color='white', 
                horizontalalignment='center', fontsize=14, weight='bold')
       plt.text(60, 95, self.vocab[top3_words][2], color='white', 
                horizontalalignment='center', fontsize=14, weight='bold')
       ax.set_axis_off()
    plt.show()