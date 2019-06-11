from utility.data_stream import text_data_writer
from utility.get_config import config
import os
import threadpool
import multiprocessing
import logging
config_dic = config("config/config.json")
logging.basicConfig(level=logging.INFO)
# wf = text_data_writer(config_dic["input_raw_train"],writer_path="data/temp_data/text_data_flow.txt")
# wf.towriter()

def writer_work(file_base, writer_path,file_name,process_id=0):
    wf = text_data_writer(file_base,writer_path,file_name,process_id)
    wf.towriter()
def main(thread_pool_num,file_base=config_dic["input_raw_train"],writer_path="data/temp_data/text_data_flow.txt"):
    file_name_list = os.listdir(file_base)
    file_num = int(len(file_name_list)/thread_pool_num)
    paramters = []
    process_pool = []
    for i in range(thread_pool_num):
        paramter = {}
        if i+1 != thread_pool_num:
            paramter["file_name"]=file_name_list[i*file_num:(i+1)*file_num]
        else:
            paramter["file_name"] = file_name_list[i*file_num:]
        paramter["writer_path"] = writer_path.split(".")[0]+str(i)+".txt"
        paramter["file_base"] = file_base
        paramter["process_id"] = i
        paramters.append((None,paramter))
        p = multiprocessing.Process(target=writer_work,kwargs=paramter)
        process_pool.append(p)
    for i in process_pool:
        i.start()
    for i in process_pool:
        i.join()


    # pool = threadpool.ThreadPool(thread_pool_num)
    # requests = threadpool.makeRequests(writer_work,paramters)
    # [pool.putRequest(req) for req in requests]
    # pool.wait()



if __name__ == '__main__':
    main(8)
    logging.info("多进程结束")
    writer_path_list = os.listdir("data/temp_data")
    wf = open("data/temp_data/text_data_flow.txt","w",encoding="utf-8")
    for path in writer_path_list:
        with open(os.path.join("data/temp_data",path),"r",encoding="utf-8") as rf:
            for line in rf:
                wf.write(line)
    wf.close()
