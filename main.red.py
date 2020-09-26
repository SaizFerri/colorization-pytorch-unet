#!/usr/bin/env python3

import json

SSH_SERVER = 'dt1.f4.htw-berlin.de'
SSH_AUTH = {'username': '{{ssh_username}}', 'password': '{{ssh_password}}'}
DATA_DIR = 'colorization-argumented/dataset_augumented_no_uncategorized'
AGENCY_URL = 'https://agency.f4.htw-berlin.de/dt'
EXPERIMENTS = [
  {
    "bins": 324,
    "lr": 0.001,
    "checkpoints_path": "colorization-argumented/checkpoints/324_small_no_uncategorized",
    "log_dir": "colorization-argumented/log/324_small_no_uncategorized",
    "divider": 2,
  },
]
# BINS = [36, 324]
# SAVED_MODEL_FILES = ['model-36-48-124.345.pth', 'model-324-43-208.819.pth']
# STEPS_PER_EPOCH = 10


batches = []

for i, experiment in enumerate(EXPERIMENTS):
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
            'dirPath': experiment["checkpoints_path"],
            'writable': True
          }
        }
      },
      # 'saved_model_dir': {
      #   'class': 'Directory',
      #   'connector': {
      #     'command': 'red-connector-ssh',
      #     'mount': True,
      #     'access': {
      #       'host': SSH_SERVER,
      #       'auth': SSH_AUTH,
      #       'dirPath': 'colorization/saved_models',
      #       'writable': True
      #     }
      #   }
      # },
      # 'saved_model_file': SAVED_MODEL_FILES[i],
      'batch_size': 64,
      'num_bins': experiment["bins"],
      'from_epoch': 0,
      'num_epochs': 50,
      'learning_rate': experiment["lr"],
      'model_divider': experiment["divider"],
      'log_dir': {
        'class': 'Directory',
        'connector': {
          'command': 'red-connector-ssh',
          'mount': True,
          'access': {
            'host': SSH_SERVER,
            'auth': SSH_AUTH,
            'dirPath': experiment["log_dir"],
            'writable': True
          }
        }
      },
      'log_file_name': 'training_'+str(experiment["bins"])+'.csv',
      'temperature': 1
    },
    'outputs': {
      'weights_file': {
        'class': 'File',
        'connector': {
          'command': 'red-connector-ssh',
          'access': {
            'host': SSH_SERVER,
            'auth': SSH_AUTH,
            'filePath': '{}/model.pth'.format(experiment["checkpoints_path"]),
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