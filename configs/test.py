import yaml 
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# print(cfg['dataset']['mean_vals'])
print(cfg['abnormalities']['both'])
print(cfg['seed'])
print(cfg['dataset']['train_dir'])