#########################################
# Project   : ARFF to CSV converter     #
# Created   : 10/01/17 11:08:06         #
# Author    : haloboy777                #
# Licence   : MIT                       #
#########################################

# Importing library
import os


# Function for converting arff list to csv list
def to_csv(text):
    data = False
    header = ""
    new_content = []
    for line in text:
        if not data:
            if "@ATTRIBUTE" in line or "@attribute" in line:
                attributes = line.split()
                if "@attribute" in line:
                    attri_case = "@attribute"
                else:
                    attri_case = "@ATTRIBUTE"
                column_name = attributes[attributes.index(attri_case) + 1]
                header = header + column_name + ","
            elif "@DATA" in line or "@data" in line:
                data = True
                header = header[:-1]
                header += "\n"
                new_content.append(header)
        else:
            new_content.append(line)
    return new_content


def convert_arrf_file_to_csv(file_directory):
    # Main loop for reading and writing files
    # Getting all the arff files from the current directory
    files = [arff for arff in os.listdir(file_directory) if arff.endswith(".arff")]
    for file in files:
        with open(f"{file_directory}/{file}") as inFile:
            content = inFile.readlines()
            name, ext = os.path.splitext(inFile.name)
            new = to_csv(content)
            with open(name + ".csv", "w") as outFile:
                outFile.writelines(new)
