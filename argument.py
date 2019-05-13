import argparse

_DATA_FOLDER = './'

def parse_arguments():
  """Parse arguments."""
  parser = argparse.ArgumentParser(
  	  description='Population random measure embedding model.')

  # data configurations
  parser.add_argument(
    '--dataset',
    default='nyt_demo',
    type=str,
    help='dataset name')

  parser.add_argument(
  	'--data_filename',
  	default=_DATA_FOLDER+'sample_data.npz',
  	type=str,
  	help='training data file name')

  parser.add_argument(
    '--vocab_filename',
    default=_DATA_FOLDER+'vocab.npz',
    type=str,
    help='vocabulary file name')

  # model configurations
  parser.add_argument(
  	'--num_topics',
  	default=20,
  	type=int,
  	help='number of topics')

  parser.add_argument(
  	'--vocab_size',
  	default=8000,
  	type=int,
  	help='vocabulary size')

  parser.add_argument(
  	'--ell_size',
  	default=20,
  	type=int,
  	help='ell latent dimension')

  parser.add_argument(
  	'--h_size',
  	default=20,
  	type=int,
  	help='h latent dimension')

  parser.add_argument(
  	'--a0',
  	default=1.0,
  	type=float,
  	help='h_n prior is Normal(0,a0*I)')

  parser.add_argument(
  	'--b0',
  	default=1.0,
  	type=float,
  	help='ell_k prior is Normal(0,b0*I)')

  parser.add_argument(
  	'--alpha0',
  	default=5.0,
  	type=float,
  	help='V_j prior is beta(1,alpha_0)')

  parser.add_argument(
  	'--gamma0',
  	default=1e3,
  	type=float,
  	help='eta_k prior is Dir([gamma_0/vocab_size])')

  parser.add_argument(
    '--beta0',
    default=5.0,
    type=float,
    help='z prior is gam(beta0*p_K,_)')

  parser.add_argument(
  	'--inference_layers',
  	default="1000,500,500",
  	type=str,
  	help='inference network layers')

  parser.add_argument(
  	'--decoder_layers',
  	default="80,120,80",
  	type=str,
  	help='decoder network layers')

  parser.add_argument(
    '--network_reg',
    default=1e-8,
    type=float,
    help='network regularization parameter')

  # training configurations
  parser.add_argument(
  	'--learning_rate',
  	default=1e-5,
  	type=float,
  	help='learning rate')

  parser.add_argument(
  	'--outer_iteration',
  	default=100,
  	type=int,
  	help='outer iteration')

  parser.add_argument(
  	'--inner_iteration',
  	default=1e3,
  	type=int,
  	help='inner iteration')
  
  args = parser.parse_args()
  return args