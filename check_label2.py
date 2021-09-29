import os
import shutil
import uuid


def Diff(li1, li2):
    return (list(list(set(li1) - set(li2)) + list(set(li2) - set(li1))))


def delete():
    image_path = '/media/vs/Data/aist/project/helmet0611/helmet0611/'
    label_path = "/media/vs/Data/aist/project/helmet0611/labels/"            #more

    image_filenames = []
    label_names = []

    for _, _, files in os.walk(label_path):
        for f in files:
            label_names.append(f.split('.')[0])

    for _, _, files in os.walk(image_path):
        for f in files:
            image_filenames.append(f.split('.')[0])
    print(label_names)
    print(image_filenames)
    print(len(image_filenames))
    print(len(label_names))

    for i in label_names:

        # print(i)
        # if i not in image_filenames:
        #     # print(1111111)
        #     img_path = os.path.join(label_path, i + '.txt')
        #     print(img_path)
        #     os.remove(img_path)
        # print(j)



        if i in image_filenames:
            # print(1111111)
            continue
        # if i not in image_filenames:
        #     print(22222)
        else:
            print(1111111)
            label_path2 = os.path.join(label_path, i + '.txt')
            print(label_path2)
            os.remove(label_path2)

    # for d in Diff(image_filenames, label_names):
    #     img_path = os.path.join(image_path, d + '.jpg')
    #     if 'train' in img_path:
    #         continue

        # os.remove(img_path)


def rename():
    img_path = '/media/xunjie/6EF7-9D71/QIMIN/0402/BK/JPEGImages/'
    label_path = "/media/xunjie/6EF7-9D71/QIMIN/0402/BK/labels/"
    # old_name -> new_name
    name_dict = dict()

    old_names = []

    for _, _, files in os.walk(img_path):
        old_names = files

    postfix_img = 'jpg'
    postfix_label = 'txt'

    for o in old_names:
        new_name = str(uuid.uuid4()) + '.' + postfix_img
        old_path = os.path.join(img_path, o)
        new_path = os.path.join(img_path, new_name)
        os.rename(old_path, new_path)
        name_dict[old_path] = new_path

    for old_name, new_name in name_dict.items():
        old_label_name = old_name.replace('.jpg', '.txt')
        old_label_name = old_label_name.replace('JEPGImages', 'labels')
        new_label_name = new_name.replace('.jpg', '.txt')
        new_label_name = new_label_name.replace('JEPGImages', 'labels')
        os.rename(old_label_name, new_label_name)


def flip(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    # print(lines)

    result = []

    for i in lines:
        if i[0] == None:
            result.append(None)
        # print(i[1])

        if i[0] == '0':
            result.append("2" + i[1:])
        elif i[1] == '0':
            continue
        elif i[0] == '1':
            result.append("1" + i[1:])
        elif i[0] == '2':
            result.append("3" + i[1:])
        elif i[0] == '3':
            result.append("4" + i[1:])


        # else:
        #     num = int(i[0])
        #     num = num + 2
        #     result.append(str(num) + i[1:])
    print(result)
    with open(txt_path, "w") as f:
        f.writelines(result)


if __name__ == "__main__":
    path = "/media/vs/Data/aist/project/TruckCover0511/labels"
    delete()

    # for _, _, files in os.walk(path):
    #     for f in files:
    #         txt_path = os.path.join(path, f)
            # flip(txt_path)
