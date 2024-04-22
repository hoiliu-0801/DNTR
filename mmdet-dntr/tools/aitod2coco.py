import os
path_old= "/mnt/data0/Garmin/datasets/ai-tod/val/labels_old/"
path_new= "/mnt/data0/Garmin/datasets/ai-tod/trainval/labels/train2024"
CLASSES = {'airplane':0, 'bridge':1, 'storage-tank':2, 'ship':3, 'swimming-pool':4, 'vehicle':5, 'person':6, 'wind-mill':7}
file_= os.listdir(path_new)
print(len(file_))
# print(file_)
# for i in file_:
#     with open(path_old+i, 'r') as f:
#         with open(path_new+i, 'w') as f2:
#             lines = f.readlines()
#             for line in lines:
#                 x1, y1, x2, y2, class_name = line.split(' ')
#                 x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
#                 w = (x2-x1)/800
#                 h = (y2-y1)/800
#                 xc = (x1+w/2)/800
#                 yc = (y1+h/2)/800
#                 if '\n' in class_name:
#                     class_name = class_name[:-1]
#                 class_new= str(CLASSES[class_name])
#                 xc, yc, w, h = str(xc), str(yc), str(w), str(h)
#                 new_line = [class_new+' '+xc+' '+yc+' '+ w +' '+ h +'\n']
#                 f2.writelines(new_line)
