import os

def rename_dataset(folders):
    for folder in folders:
        count = 1
        for filename in os.listdir(folder):
            os.renames(folder + "/" + filename, folder + "/" + str(count) + ".jpg")
            count = count + 1

folders = ["./dataset/mixed/training_set/AI_generated",
           "./dataset/mixed/training_set/hand_drawing",
           "./dataset/mixed/testing_set/AI_generated",
           "./dataset/mixed/testing_set/hand_drawing"]

rename_dataset(folders)
