#!/usr/bin/env python3

import json

SSH_SERVER = 'dt1.f4.htw-berlin.de'
SSH_AUTH = {'username': '{{ssh_username}}', 'password': '{{ssh_password}}'}
DATA_DIR = 'colorization/dataset'
AGENCY_URL = 'https://agency.f4.htw-berlin.de/dt'
BINS = [36, 324]
# STEPS_PER_EPOCH = 10


batches = []

for i, _bins in enumerate(BINS):
  batch = {
    'inputs': {
      'data_dir': {
        'class': 'Directory',
        'connector': {
          'command': 'red-connector-ssh',
          'mount': True,
          'access': {
            'host': SSH_SERVER,
            'auth': SSH_AUTH,
            'dirPath': DATA_DIR
          }
        }
      },
      'checkpoints_dir': {
        'class': 'Directory',
        'connector': {
          'command': 'red-connector-ssh',
          'mount': True,
          'access': {
            'host': SSH_SERVER,
            'auth': SSH_AUTH,
            'dirPath': 'colorization/checkpoints',
            'writable': True
          }
        }
      },
      'batch_size': 64,
      'num_bins': _bins,
      'from_epoch': 0,
      'num_epochs': 50,
      # 'learning_rate': learning_rate,
      'log_dir': {
        'class': 'Directory',
        'connector': {
          'command': 'red-connector-ssh',
          'mount': True,
          'access': {
            'host': SSH_SERVER,
            'auth': SSH_AUTH,
            'dirPath': 'colorization/log',
            'writable': True
          }
        }
      },
      'log_file_name': 'training_'+str(_bins)+'.log'
    },
    'outputs': {
      'weights_file': {
        'class': 'File',
        'connector': {
          'command': 'red-connector-ssh',
          'access': {
            'host': SSH_SERVER,
            'auth': SSH_AUTH,
            'filePath': 'checkpoints/model.pth',
          }
        }
      }
    }
  }
  batches.append(batch)

with open('main.cwl.json') as f:
  cli = json.load(f)

red = {
  'redVersion': '9',
  'cli': cli,
  'batches': batches,
  'container': {
    'engine': 'docker',
    'settings': {
      'image': {
        'url': 'saizferri/colorization-u-net',
      },
      'ram': 60000,
      'gpus': {
        'vendor': 'nvidia',
        'count': 1
      }
    }
  },
  'execution': {
      'engine': 'ccagency',
      'settings': {
          'access': {
            'url': AGENCY_URL,
            'auth': {
                'username': '{{agency_username}}',
                'password': '{{agency_password}}'
            }
          }
      }
  }
}

with open('main.red.json', 'w') as f:
  json.dump(red, f, indent=4)