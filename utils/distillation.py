import torch
import torch.nn as nn
import tqdm
from losses.metrics import F1Score


class Distiller:
    def __init__(self, teacher, student, temperature=3, alpha=0.9, lr=1e-4):
        self.teacher = teacher
        self.student = student

        self.temperature = temperature
        self.alpha = alpha
        self.optim = torch.optim.Adam(self.student.parameters(), lr)

        self.student_loss = nn.CrossEntropyLoss(reduction='mean')
        self.distillation_loss = nn.KLDivLoss(reduction='batchmean')
        self.metrics_dict = {"f1_score": F1Score()}

    def train_step(self, x, y):
        with torch.no_grad():
            teacher_out = self.teacher(x)
        student_out = self.student(x)
        student_loss = self.student_loss(student_out, y)
        distill_loss = self.distillation_loss(torch.softmax(student_out / self.temperature, 1).log(),
                                              torch.softmax(teacher_out / self.temperature, 1))
        loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return loss

    def train_single_epoch(self, dataloader, epoch):
        postfix_dict = {}
        self.teacher.eval()
        self.student.train()

        total_step = len(dataloader)
        total_loss = 0
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, position=0, leave=False)
        for i, (images, targets) in tbar:
            images = images.cuda()
            targets = targets.cuda()
            loss = self.train_step(images, targets)
            total_loss += loss.item()
            postfix_dict['train/loss'] = loss.item()

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format('train')
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)
        total_loss /= len(dataloader)
        return total_loss

    def evaluale_single_epoch(self, dataloader, epoch, metrics_dict):
        postfix_dict = {}
        self.student.eval()
        with torch.no_grad():
            total_step = len(dataloader)

            loss_list = []
            tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, position=0, leave=False)
            for i, (images, targets) in tbar:
                images = images.cuda()
                targets = targets.cuda()
                outputs = self.student(images)
                loss = self.student_loss(outputs, targets)
                loss_list.append(loss.item())

                outputs = outputs.cpu().numpy()
                targets = targets.cpu().numpy()

                for _, metric_inst in metrics_dict.items():
                    metric_inst.update(outputs, targets)

                f_epoch = epoch + i / total_step
                desc = '{:5s}'.format('val')
                desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
                tbar.set_description(desc)
                tbar.set_postfix(**postfix_dict)
            return sum(loss_list) / len(loss_list)