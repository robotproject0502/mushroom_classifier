from CNN import EfficientNet
from datetime import datetime
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn

from utils.image_loader import load_data
from utils.config import Config
from utils.model_util import validation
from utils.model_util import draw_plot

def train():
    config = Config().params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data and set model for fine-tuning
    train_loader, vad_loader = load_data(config)

    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
    model = EfficientNet.from_name('efficientnet-b0')
    model.load_state_dict(torch.load(config["model_load_path"], map_location=device))
    model._fc = nn.Linear(1280, 2)

    for name, param in model.named_parameters():
        if '_fc' in name:
            continue
        param.requires_grad = False
    model = model.to(device)

    # set loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.98)

    train_accs, vad_accs = [], []
    now = datetime.now()
    name = '{:%mM %dD %H %MM}'.format(now)
    print(len(train_loader))

    for epoch in range(config["epoch"]):
        print('Epoch : {}'.format(epoch))
        train_loss = 0
        train_acc = 0
        scheduler.step()

        torch.save(model.state_dict(), config['model_save_path'] + "\{} {}E.tar".format(name, epoch))

        for step, (images, labels) in enumerate(train_loader):
            if step % 50 == 0:
                print('Step   : {}'.format(step))
            # set data and predicted
            images = Variable(images, requires_grad=True).to(device)
            labels = Variable(labels).to(device)
            outputs = model(images)

            # train model by difference
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            outputs = torch.argmax(outputs, dim=1)
            train_acc += (outputs == labels).float().mean()
            train_loss += loss.item()

            # development test and print loss, acc
            if step == len(train_loader) - 1:
                model.eval()
                with torch.no_grad():
                    train_loss = train_loss / len(train_loader)
                    train_acc = train_acc / len(train_loader)
                    vad_loss, vad_acc = validation(model, vad_loader, criterion, device)
                print("Epoch: {}/{}..".format(epoch+1, config["epoch"]),
                      "Train Loss: {:.6f}..".format(train_loss),
                      "Train Acc: {:.6f}..".format(train_acc),
                      "Vad Loss: {:.6f}..".format(vad_loss),
                      "Vad Acc: {:.6f}".format(vad_acc))
                train_accs.append(train_acc)
                vad_accs.append(vad_acc)
                model.train()

    # after train save model and acc graph
    torch.save(model.state_dict(), config['model_save_path'] + "\{} {}E.tar".format(name, epoch))
    draw_plot(train_accs, vad_accs, name, config)

if __name__ == '__main__':
    train()
