import torchvision.models as models
import torchvision.io as io
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dataset_path = "./dataset/ganyu"
training_set_size = len(os.listdir(dataset_path + "/training_set/AI_generated"))
testing_set_size = len(os.listdir(dataset_path + "/testing_set/AI_generated"))
num_classes = 2
batch_size = 50

training_images = []
training_labels = []
testing_images = []
testing_labels = []
loss_array = []
accuracy_array = []


# read training data
for i in range(training_set_size):
    image = io.read_image(path=dataset_path + "/training_set/AI_generated/" + str(i + 1) + ".jpg")
    training_images.append(transforms.Resize((200, 200))(image.float()))
    training_labels.append(nn.functional.one_hot(torch.tensor(1), num_classes))
    image = io.read_image(path=dataset_path + "/training_set/hand_drawing/" + str(i + 1) + ".jpg")
    training_images.append(transforms.Resize((200, 200))(image.float()))
    training_labels.append(nn.functional.one_hot(torch.tensor(0), num_classes))

# read testing data
for i in range(testing_set_size):
    image = io.read_image(path=dataset_path + "/testing_set/AI_generated/" + str(i + 1) + ".jpg")
    testing_images.append(transforms.Resize((200, 200))(image.float()))
    testing_labels.append(nn.functional.one_hot(torch.tensor(1), num_classes))
    image = io.read_image(path=dataset_path + "/testing_set/hand_drawing/" + str(i + 1) + ".jpg")
    testing_images.append(transforms.Resize((200, 200))(image.float()))
    testing_labels.append(nn.functional.one_hot(torch.tensor(0), num_classes))

model = models.googlenet(weights='DEFAULT', progress=True)
model.fc = nn.Sequential(
    nn.Linear(1024, num_classes),
    nn.Softmax())

model.train()

transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

loss_function = nn.CrossEntropyLoss()

def compute_accuracy(output, label):
    count = 0
    for i in range(len(output)):
        if torch.argmax(output[i], dim=0) == torch.argmax(label[i], dim=0):
            count = count + 1

    return count / len(output)

def train_model(epoch, optimizer):
    turns = 0
    for i in range(epoch):
        for j in range(int(training_set_size * 2 / batch_size)):
            turns = turns + 1
            batch_input_image = torch.zeros(batch_size, 3, 200, 200)
            batch_label = torch.zeros(batch_size, 2)

            for k in range(batch_size):
                batch_input_image[k] = transform(training_images[j * batch_size + k])
                batch_label[k] = training_labels[j * batch_size + k]

            optimizer.zero_grad()
            output = model(batch_input_image)
            loss = loss_function(batch_label, output)
            loss.backward()
            optimizer.step()
            accuracy = compute_accuracy(output, batch_label)

            loss_array.append(loss.item())
            accuracy_array.append(accuracy)
            print("turns : %d, loss : %f, accuracy: %f" % (turns, loss, accuracy))

def model_test():
    test_input_image = torch.zeros(len(testing_images), 3, 200, 200)
    test_label = torch.zeros(len(testing_labels), 2)

    for k in range(len(testing_labels)):
        test_input_image[k] = transform(testing_images[k])
        test_label[k] = testing_labels[k]

    output = model(test_input_image)
    print("test accuracy is %f" % (compute_accuracy(output, test_label)))

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_model(75, optimizer)
model_test()

plt.plot(loss_array)
plt.title('Fine-tuning GoogleNet model on Ganyu image set')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
plt.plot(accuracy_array)
plt.title('Fine-tuning GoogleNet model on Ganyu image set')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()