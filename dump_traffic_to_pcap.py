# Step 1: Import packages and set the working directory
import subprocess as sub
import os
from flowmeter import Flowmeter
from rich import print
from rich.theme import Theme
from rich.console import Console

# dict of rich colors
# color used in project
ct = Theme({
    'good': "bold green ",
    'bad': "red",
    'blue': "blue",
    'yellow': "yellow",
    'purple': "purple",
    'magenta': "magenta",
    'cyan': "cyan"
})
rc = Console(record=True, theme=ct)


# Step2: Include files for logo, dashboard and task manager

# os.system(
#     'python3 rich/logo.py')

# os.system(
#     'python3 rich/task-analyzer.py')

# os.system(
#     'python3 rich/cli_dashboard.py')
# # 
# os.system(
#     'python3 rich/logo.py')

# Step 3: Capture network traffic and dump in to a *.pcap file formate

file = 0

# while(file <= 5):
while True:
    file_name = file + 1

    rc.log(
        "\n[cyan][* ] - Dumping file{} as data{}.pcap[/]\n".format(file_name, file_name))

    rc.log(
        "\n[blue]<--------------------File {}------------------------>[/]\n".format(file_name))

    p = sub.Popen(('sudo dumpcap', '-i', 'wlp0s20f3', '-a', 'filesize:10',
                   '-w', 'pcapF/data{}.pcap'.format(file_name)), stdout=sub.PIPE)

    for row in iter(p.stdout.readline, b''):
        rc.log(row.rstrip())   # process here

    rc.log(
        "\n[good][ DONE ][/][cyan] - Saved pcap file as data{}.pcap[/]\n".format(file_name))

    # Convert the *.pcap to *.csv file
    # os.system(
    #     'python3 pcap_to_csv.py')
    # with open("csvs/merged_data.csv", "w") as my_empty_csv:
    #     # now you have an empty file already
    #     pass 
    path = 'pcapF/'
    data_pcap = 'pcapF/data{}.pcap'.format(file_name)

    arr = os.listdir(path)
    cat_pcap = path + arr[file]
    print(cat_pcap, 'for this filw')
    feature_gen = Flowmeter(offline=data_pcap, outfunc=None,
                            outfile='csvs/merged_data.csv')
    feature_gen.run()
    
 
    

    # Normalize and preprocessing of *.csv file to fed the ML/ DL models.
    os.system(
        'python3 norrm.py')

    # classify the activity from model
    os.system(
        'python3 model.py')
    g = 'pcapF/data{}.pcap'.format(file_name)
    f = 'csvs/out{}.csv'.format(file_name)
    gg = 'csvs/merged_data.csv'
    # os.remove(gg)
    print(data_pcap, 'cat')
    


    file += 1

rc.save_html("report.html")
rc.log('exit')
