import argparse

#============================================================================================

parser = argparse.ArgumentParser(description='')
#Training Setting
parser.add_argument('--screen_w', dest='screen_w', default=1200, help='input_x width')
parser.add_argument('--screen_h', dest='screen_h', default=650, help='input_x height')
parser.add_argument('--channel_dim', dest='channel_dim', default=3, help='channel dimension')
parser.add_argument('--action_num', dest='action_num', default=6, help='number of actions')

parser.add_argument('--epsilon', dest='epsilon', default=0.1, help='epsilon rate')

parser.add_argument('--batch_size', dest='batch_size', default=128, help='batch size')
parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, help='learning rate')

config = parser.parse_args()
#============================================================================================

# def get_config():
#     return args