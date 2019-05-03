import os
import re
import collections
from datetime import datetime

class flow():
    def __init__(self):
        self.data_flow = collections.defaultdict(int)
        self.hour_flow = collections.defaultdict(int)
        self.week_flow = collections.defaultdict(int)           #星期即的人流量
class data():
    def __init__(self,picture_id,label,people_data):
        self.picture_id = picture_id
        self.lable = label
        self.people_data = people_data
        self.flow = flow()
        self.updata_flow()
    def updata_flow(self):
        for people in self.people_data:
            for date,hour in people.time.items():
                self.flow.data_flow[date] += 1
                self.flow.week_flow[datetime.strptime(date,"%Y%m%d").weekday()] += 1
                for one_hour in hour:
                    self.flow.hour_flow[one_hour] += 1


class people_data():
    def __init__(self,people_id,time):
        self.people_id = people_id
        self.time = time #{莫一天：小时}
    def _get_hold_len(self):
        "拿到给人在该区域的驻留时间"
        pass

class from_text_get_data_util():
    @staticmethod
    def get_all_picture_people_data(file_base):
        file_name_list = os.listdir(file_base)
        all_data_lt = []
        for file_name in file_name_list:
            file_name_data = file_name[:-4]
            temp = file_name_data.split("_")
            picture_id = int(temp[0])
            label = int(temp[1])
            file_name = os.path.join(file_base,file_name)
            people_list = []
            with open(file_name,"r",encoding="utf-8") as rf:
                for line in rf:
                    temp = line.strip().split("\t")
                    time = {}
                    for data_hour in temp[1].split(","):
                        inter_temp = data_hour.split("&")
                        time[inter_temp[0]] = inter_temp[1].split("|")
                    people_list.append(people_data(temp[0],time))
            all_data_lt.append(data(picture_id,label,people_list))
        return all_data_lt
if __name__ == '__main__':
    temp = from_text_get_data_util.get_all_picture_people_data("../data")
    print("test")