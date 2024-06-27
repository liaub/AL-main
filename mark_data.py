import numpy as np
import matplotlib.pyplot as plt
from config import args
import os
import pandas as pd
'''
Drift type：0.sudden; 1.gradual; 2.incremental; 3.normal
Drift point: [0, 49]
'''

all_list = os.listdir(args.DATA_FILE + 'mark')
# retrieve all untagged original_data.
need_lables_file = [dir for dir in all_list if '_.csv' in dir]
for file in need_lables_file:
    labels_reason = "["
    input("Start tag "+file+", press Enter to continue.")
    df = pd.read_csv(args.DATA_FILE + 'mark/' + file)
    data = df.iloc[:, -1].tolist()
    print("Tag original_data source, press Enter to continue：", data)

    print("Data stream image display, you need to consider the drift type [0,1,2,3] and drift point [0,49] of the image, then close the image before tagging.")
    x = np.arange(1, len(data)+1)
    y = np.array([abs(d) for d in data])
    # y = original_data
    plt.title("Matplotlib demo")
    plt.xlabel("x axis caption")
    plt.ylabel("y axis caption")
    plt.plot(x, y)
    plt.show()
    y = input("Please enter the drift category of the image（0,1,2,3）, press Enter to continue:")
    labels_reason = labels_reason+str(y)+","
    loc_y = input("Please enter the drift point of the image [0,49], press Enter to continue:")
    labels_reason = labels_reason + str(loc_y) + "]"

    # resave as CSV
    p = labels_reason+'.csv'
    new_file = str(file).replace('_.csv', p)
    tag_label_path = args.DATA_FILE + 'mark/' + new_file
    df = pd.DataFrame(data)
    # save dataframe
    df.to_csv(tag_label_path)
    print(file+"Data tagging complete.")


