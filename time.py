import os
import yaml 
import datetime
# Load the Config File
with open("./configs/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

logs_dir = cfg['logs_path'] + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
# time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
# mydir = os.path.join(logspath, time)
name = "save.png"
print(f"{logs_dir}/{name}")
# if not os.path.exists(logs_dir):
#     os.makedirs(logs_dir)