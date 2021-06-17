import torch 


if __name__=="__main__":
    x = torch.rand(2,3).cuda()
    y = x * 2
    print(y)
    y = y + 1.0
    print("ahihi")