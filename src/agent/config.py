import argparse

#============================================================================================

parser = argparse.ArgumentParser(description='')

parser.add_argument('--modelpath', dest='modelpath', default='./agent/model/', help='model path')
parser.add_argument('--graphpath', dest='graphpath', default='./agent/graph', help='log path')


#Training Setting
parser.add_argument('--screen_w', dest='screen_w', default=128, help='input_x width')
parser.add_argument('--screen_h', dest='screen_h', default=75, help='input_x height')
parser.add_argument('--channel_dim', dest='channel_dim', default=3, help='channel dimension')
parser.add_argument('--action_num', dest='action_num', default=3, help='number of actions')

parser.add_argument('--epsilon', dest='epsilon', default=0.10, help='epsilon rate')

parser.add_argument('--batch_size', dest='batch_size', default=400, help='batch size')
# parser.add_argument('--ep_per_batch', dest='ep_per_batch', default=200, help='episode per batch')
parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, help='learning rate')
parser.add_argument('--momentum', dest='momentum', default=0.50, help='momentum')

parser.add_argument('--checkpoint', dest='checkpoint', default=50, help='checkpoint')
parser.add_argument('--continue_training', dest='continue_training', default=False, help='flag to see whether to start new training')
config = parser.parse_args()
#============================================================================================

# def get_config():
#     return args