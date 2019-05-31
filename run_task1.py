from utility.data_stream import text_data_writer
from utility.get_config import config
config_dic = config("config/config.json")
wf = text_data_writer(config_dic["input_raw_train"],writer_path="data/temp_data/text_data_flow.txt")
wf.towriter()
