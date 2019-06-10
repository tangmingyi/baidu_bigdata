import cv2
from lib.resnet import ResNet152
import paddle
from paddle import fluid
import shutil
import os
import json
from visualdl import LogWriter
import numpy as np
config = json.load(open("config/config.json","r",encoding="utf-8"))
# img = cv2.imread("D:\\programing_data\\train\\picture\\000009_001.jpg")
logw = LogWriter(config["log_output"],sync_cycle=10)
def creat_train_reader(pic_file,flow_file):
    def train_reader():
        with open(flow_file,"r",encoding="utf-8") as rf:
            for line in rf:
                flow = json.loads(line.strip())
                pic_name = flow["picture_id"]+"_"+flow["label"]+".jpg"
                pic_id,pic_label = pic_name.split(".")[0].split("_")
                path = os.path.join(os.path.join(pic_file,pic_label),pic_name)
                img = cv2.imread(path)
                if type(img) == type(None):
                    print("worry with path:%s"%path)
                    continue
                # img = np.reshape(img, [3, 100, 100])
                img = img.flatten()
                yield (img,int(pic_label)-1)
    return train_reader
# for i in creat_train_reader("D:\\programing_data\\train\\picture","data/temp_data/text_data_flow.txt")():
#     print(i)

train_reader = paddle.batch(creat_train_reader(config["input_picture_train"],"data/temp_data/text_data_flow.txt"),config["train_batch_size"])

pic_input = fluid.layers.data(name='image',shape=[3,100,100],dtype='float32')
# flow_input = fluid.layers.data(name='text',shape=[100],dtype='float32')
label = fluid.layers.data(name="label",shape=[1],dtype="int64")

pic_res_net = ResNet152()
pic_tensor = pic_res_net.net(pic_input,9)
# fina_lay = fluid.layers.fc()
cost=fluid.layers.softmax_with_cross_entropy(logits=pic_tensor,label=label)
avg_cost=fluid.layers.mean(cost)
acc=fluid.layers.accuracy(input=pic_tensor,label=label)

optimizer=fluid.optimizer.AdamOptimizer(learning_rate=config["learning_rate"])
opts=optimizer.minimize(avg_cost)
place_cpu = fluid.CPUPlace()
place_gpu = fluid.CUDAPlace(0)
exe = fluid.Executor(place=place_gpu)
start_up_program = fluid.default_startup_program()
temp_vars = start_up_program.global_block().vars
vars_list = []
log_list = []
vars_list.append(avg_cost)
vars_list.append(acc)
with logw.mode("train") as writer:
    log_list.append(writer.scalar("loss"))
    log_list.append(writer.scalar("acc"))
for k,v in fluid.default_startup_program().global_block().vars.items():
    if k[-7:] == "weights":
        vars_list.append(v)
        with logw.mode("train") as writer:
            log_list.append(writer.histogram(v.name,100))
exe.run(fluid.default_startup_program())
if os.path.exists(config["res_net_model"]):
    print("初始化模型参数 path：%s"%config["res_net_model"])
    fluid.io.load_params(executor=exe,dirname=config["res_net_model"])

feeder = fluid.DataFeeder(place=place_cpu,feed_list=[pic_input,label])
for i in range(config["epoch"]):
    print("********")
    print("epoch %s"%i)
    for index,data in enumerate(train_reader()):
        # loss,myacc,output
        run_list = exe.run(program=fluid.default_main_program(),feed=feeder.feed(data),fetch_list=vars_list)
        if (index+1)%100==0:
            print("*******")
            print("step:%s"%(index+1))
            print(run_list[0])
            print(run_list[1])
            log_list[0].add_record(index,run_list[0])
            log_list[1].add_record(index,run_list[1])
            for num,log in enumerate(log_list[2:]):
                log.add_record(index,run_list[num+2].flatten())


        if (index+1)%config["save_model_step"]==0:
        # if index == 10:
            shutil.rmtree(config["res_net_model"], ignore_errors=True)
            if not os.path.exists(config["res_net_model"]):
                os.makedirs(config["res_net_model"])
            fluid.io.save_params(executor=exe,dirname=config["res_net_model"])



