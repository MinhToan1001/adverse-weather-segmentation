import torch
from src.evaluation.metrics import confusion_matrix,scores

def test_ignore_label():
    pred=torch.tensor([[0,1],[2,0]])
    true=torch.tensor([[0,1],[255,2]])
    matrix=confusion_matrix(pred,true,3)
    assert int(matrix.sum())==3
    assert abs(scores(matrix)['miou']-(0.5+1+0)/3)<1e-6
