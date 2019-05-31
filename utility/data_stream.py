import os
import collections
from datetime import datetime
import json
import matplotlib.pyplot as plt
import logging


class flow():
    def __init__(self):
        self.data_flow = collections.defaultdict(int)
        self.hour_flow = collections.defaultdict(int)
        self.week_flow = collections.defaultdict(int) # 星期即的人流量


class text_data():
    def __init__(self, picture_id, label, people_data):
        self.picture_id = picture_id
        self.lable = label
        self.people_data = people_data
        self.flow = flow()
        self._updata_flow()

    def _updata_flow(self):
        #todo:可能会有一些离群点需要清洗。
        week_flow = collections.defaultdict(int)
        hour_flow = collections.defaultdict(int)
        for people in self.people_data:
            for date, hour in people.time.items():
                self.flow.data_flow[str(date)] += 1
                week_flow[date + "_" +str(datetime.strptime(date, "%Y%m%d").weekday())] += 1
                for one_hour in hour:
                    hour_flow[date + "_" +str(one_hour)] += 1
        for date,num in week_flow.items():
            if self.flow.week_flow[date.split("_")[-1]] == 0:
                self.flow.week_flow[date.split("_")[-1]] += num
            else:
                # try:
                self.flow.week_flow[date.split("_")[-1]] = float(self.flow.week_flow[date.split("_")[-1]] + num)/2.0
                # except Exception as e:
                #     print(e)
        for date,num in hour_flow.items():
            if self.flow.hour_flow[date.split("_")[-1]] == 0:
                self.flow.hour_flow[date.split("_")[-1]] += num
            else:
                self.flow.hour_flow[date.split("_")[-1]] = float(self.flow.hour_flow[date.split("_")[-1]] + num)/2.0


    def flow_view(self):
        def get_flow_tuple(input_dic):
            """
            :param input_dic:
            :return: x_list,y_list(height)
            """
            tup_lt = []
            for x,y in input_dic.items():
                tup_lt.append((x,y))
            tup_lt = sorted(tup_lt,key=lambda x:x[0])
            return ([x[0] for x in tup_lt],[x[1] for x in tup_lt])

        data = get_flow_tuple(self.flow.data_flow)
        hour = get_flow_tuple(self.flow.hour_flow)
        week = get_flow_tuple(self.flow.week_flow)
        fig = plt.figure(figsize=(20,15))
        ax = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)
        ax.set_xlabel("time")
        ax.set_ylabel("flow")
        ax.bar(x=data[0],height=data[1])
        # plt.xticks(rotation=90)
        ax1.bar(x=hour[0],height=hour[1],width=0.8)
        plt.xticks(rotation=90)
        ax2.bar(x=week[0],height=week[1])
        # plt.xticks(rotation=90)
        plt.show()



class people_data():
    def __init__(self, people_id, time):
        self.people_id = people_id
        self.time = time  # {莫一天：小时}

    def _get_hold_len(self):
        "拿到给人在该区域的驻留时间"
        pass


class from_text_get_data():
    @staticmethod
    def get_all_picture_people_data(file_base,show_view=False):
        "这是一个生成器，用于生成data"
        file_name_list = os.listdir(file_base)
        # all_data_lt = []
        for file_name in file_name_list:
            file_name_data = file_name[:-4]
            temp = file_name_data.split("_")
            picture_id = temp[0]
            label = temp[1]
            file_name = os.path.join(file_base, file_name)
            people_list = []
            with open(file_name, "r", encoding="utf-8") as rf:
                for line in rf:
                    temp = line.strip().split("\t")
                    time = {}
                    for data_hour in temp[1].split(","):
                        inter_temp = data_hour.split("&")
                        time[inter_temp[0]] = inter_temp[1].split("|")
                    people_list.append(people_data(temp[0], time))
            # try:
            output = text_data(picture_id, label, people_list)
            # except Exception as e:
            #     print(e)
            if show_view:
                output.flow_view()
            yield output


class text_data_writer():
    def __init__(self, file_base, writer_path):
        self.data_genrater = from_text_get_data.get_all_picture_people_data(file_base,show_view=False)
        self.wf = open(writer_path, "w", encoding="utf-8")


    def towriter(self):
        for index,data in enumerate(self.data_genrater):
            if index % 100 == 0:
                logging.info("has precess {}".format(index))
            self.wf.write(json.dumps(
                {"picture_id":data.picture_id,"label":data.lable,"week_flow": data.flow.week_flow, "hour_flow": data.flow.hour_flow, "data_flow": data.flow.data_flow}))
            self.wf.write("\n")




