import os

def time_trans(input):
    h, m, s = input.split(":")
    output = int(h) * 3600 + int(m) * 60 + int(s)
    return output


with open("out_trajectory.txt", "r") as rfp:
    lines = rfp.readlines()
os.remove("out_trajectory.txt")
final_line = []
with open("out_trajectory.txt", "w") as wfp:
    last_line = ""
    record_line = []
    for line in lines:
        line = line.strip()
        split_line = line.split(",")
        video_source = split_line[0]
        time = time_trans(split_line[1])
        pos = split_line[2:4]
        conf = split_line[4]
        if last_line == "":
            record_line.append(split_line)
            last_line = split_line
        elif time - time_trans(last_line[1]) == 2 and video_source == last_line[0]:
            record_line.append(split_line)
            last_line = split_line
        else:
            flag = 0
            xieru = record_line[0][0] + "," + record_line[0][1] + "-" + record_line[-1][1] + ","
            for rec_split_line in record_line:
                if flag != len(record_line) - 1:
                    xieru += rec_split_line[2] + "," + rec_split_line[3] + ","
                else:
                    xieru += rec_split_line[2] + "," + rec_split_line[3]
                flag += 1
            final_line.append(xieru)
            record_line = []
            record_line.append(split_line)
            last_line = split_line
    flag = 0
    xieru = record_line[0][0] + "," + record_line[0][1] + "-" + record_line[-1][1] + ","
    for rec_split_line in record_line:
        if flag != len(record_line) - 1:
            xieru += rec_split_line[2] + "," + rec_split_line[3] + ","
        else:
            xieru += rec_split_line[2] + "," + rec_split_line[3]
        flag += 1
    final_line.append(xieru)
    final_line.sort(key= lambda x: time_trans(x.split(",")[1].split("-")[0]))
    for xieru in final_line:
        wfp.write(xieru)
        wfp.write("\n")






