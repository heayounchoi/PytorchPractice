import torch

lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())

bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())
