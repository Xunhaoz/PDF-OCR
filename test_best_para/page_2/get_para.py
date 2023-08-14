from pprint import pprint

lines = []

with open("page_2.txt") as f:
    lines = f.readlines()

if len(lines) % 2 == 1:
    lines = lines[:-1]

all_list = []
for index in range(0, len(lines), 2):
    factor = lines[index].replace("\n", "").split(': ')[1]
    factor = list(map(float, factor[1:-1].split(", ")))

    ac_rate = lines[index + 1].replace("\n", "").split(': ')[1]
    ac_rate = float(ac_rate)

    tmp_map = {
        "factor": factor,
        "ac_rate": ac_rate
    }

    all_list.append(tmp_map)

all_list.sort(key=lambda x: x['ac_rate'])
pprint(all_list)
