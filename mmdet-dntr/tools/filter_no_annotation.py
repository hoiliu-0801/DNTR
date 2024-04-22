import json
# JSON file
resFile = '/mnt/data0/Garmin/datasets/ai-tod/annotations/aitod_test_v1_new.json'
f = open (resFile, "r")
anns = json.loads(f.read())
anns = anns['images']

def Merge(dict_1, dict_2):
	result = dict_1 | dict_2
	return result

merge={}
for ann in anns:
    a={ann['file_name'][:-4]:ann['id']}
    merge.update(a)

# Driver code
# dict_1 = {'John': 15, 'Rick': 10, 'Misa' : 12 }
# dict_2 = {'Bonnie': 18,'Rick': 20,'Matt' : 16 }
# dict_3 = Merge(dict_1, dict_2)
# print(dict_3)

# annsImgIds = [ann['file_name'][:-4] for ann in anns]
# a = set(annsImgIds)
# {'area': 176, 'bbox': [618, 484, 22, 8], 'category_id': 5, 'id': 21218, 'image_id': 882, 'iscrowd': 0, 'segmentation': []}
# [{"file_name": "22766.png", "id": 0, "width": 800, "height": 800}, {"file_name": "2053__2298_1200.png", "id": 1, "width": 800, "height": 800}
resFile_2 = '/mnt/data0/Garmin/ultralytics/runs/detect/train72/predictions.json'
f2 = open (resFile_2, "r")
anns_2 = json.loads(f2.read())
for ann2 in anns_2:
    ann2['image_id'] = merge.get(ann2['image_id'],0)
    # if merge.get(ann2['image_id'])==None:
    #      print(ann2['image_id'])
    #      exit()
# print(anns_2[0])
with open('/mnt/data0/Garmin/ultralytics/runs/detect/train72/predictions_8s.json', 'w') as f:
    json.dump(anns_2, f)
#  {'image_id': '9999984_00000_d_0000160__0_0', 'category_id': 5, 'bbox': [333.48, 494.464, 69.869, 34.831], 'score': 0.01973}
exit()
annsImgIds_2 = [ann2['image_id'] for ann2 in anns_2]
b = set(annsImgIds_2)
print(len(a))
print(len(b))
print(len(a&b))
print(len(a-b))