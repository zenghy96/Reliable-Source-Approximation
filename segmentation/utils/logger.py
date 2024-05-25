import json
import os
import time
import sys
import torch
import tensorboardX


class Logger:
    def __init__(self, args):
        """Create a summary writer logging to log_dir."""
        log_dir = log_dir = f"{args.save_dir}/log"
        self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        self.log = open(log_dir + '/train_log.txt', 'w')
        self.start_line = True
        # write config
        para = dict((name, getattr(args, name)) for name in dir(args) if not name.startswith('_'))
        file_name = os.path.join(log_dir, 'args.txt')
        with open(file_name, 'wt') as f:
            json.dump(para, f, indent=4)

    def write(self, txt):
        self.log.write(txt)
            
    def close(self):
        self.log.close()
    
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)