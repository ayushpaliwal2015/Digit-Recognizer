import configparser

config_file_path = "./config.ini"

# function to read config file settings
def read_config():
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return dict(config)
