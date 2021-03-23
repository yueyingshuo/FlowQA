import torch
question=[torch.LongTensor([    2, 18412, 18311, 18365, 18606, 18388,     1, 18856,     3]),
       torch.LongTensor([    2, 19259, 18388, 19471, 18412, 18335, 18561, 18856,     3]),
       torch.LongTensor([    2, 20267, 18294, 19150, 33289, 18756, 18683, 18305, 18294, 19285,
        20058, 10604, 18856,     3]),
        torch.LongTensor([2, 3434, 18294, 19150, 33289, 18756, 18683, 18305, 18294, 19285,
                            20058, 10604, 18856, 3])
          ]
#doc_hiddens [33,583,250] 变成[3,583,250]  [4,14]变成[2,14]

question_len=max(len(w) for w in question)
input=torch.LongTensor(4,question_len).fill_(0)
for i,doc in enumerate(question):
    select_len=min(question_len,len(doc))
    input[i,:select_len]=doc

# mean=torch.add(input[0],input[1])/2
#
# doc_hiddens=torch.split(input,1,dim=0)
#
bs=2
list=[]
p=torch.LongTensor(2,14)
for batch_i in range(2):
    i=bs * batch_i
    tmp=input[i]
    while True:
        tmp=torch.add(tmp,input[i+1])
        i += 1
        if (i+1)>=bs*(batch_i+1):
            break
    p[batch_i,]=tmp


doc=torch.cat(list,dim=1)
print('question')