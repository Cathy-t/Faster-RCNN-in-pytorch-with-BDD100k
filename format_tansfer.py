
import json
path = r"D:\learning\机器学习与大数据分析\project\LAB4\data\bdd100k\labels\bdd100k_labels_images_val.json"
transfer_dict = []
with open(path, 'r') as load_f:
    load_dict = json.load(load_f)

for dict in load_dict:
    name = dict['name']
    # print(dict)
    # print(name)
    for object in dict['labels']:
        # print(object)
        keys = object.keys()
        if 'box2d' not in keys:
            continue
        category = object['category']
        box2d = object['box2d']
        bbox = []
        bbox.append(box2d["x1"])
        bbox.append(box2d["y1"])
        bbox.append(box2d["x2"])
        bbox.append(box2d["y2"])

        jsontext = {
            'name': name,
            'timestamp': 1000,
            'category': category,
            'bbox': bbox
        }

        transfer_dict.append(jsontext)

after_path = r"D:\learning\机器学习与大数据分析\project\LAB4\data\bdd100k\labels\gt_val.json"
with open(after_path, "w") as dump_f:
    json.dump(transfer_dict,dump_f)


